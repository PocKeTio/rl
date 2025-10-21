"""
Proximal Policy Optimization (PPO) avec AMP (Automatic Mixed Precision).

Implémente:
- PPO avec clipping
- GAE pour les advantages
- Loss policy + value + entropy
- Gradient clipping
- AMP pour optimiser la VRAM
- Annealing de l'entropy coefficient
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..models.policy_lstm import PolicyLSTM
from .storage import RolloutStorage


class PPO:
    """
    Proximal Policy Optimization avec support LSTM.
    """
    
    def __init__(
        self,
        policy: PolicyLSTM,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        use_amp: bool = True,
        target_kl: Optional[float] = None,
        device: str = "cuda",
    ):
        """
        Args:
            policy: Réseau de policy LSTM
            learning_rate: Learning rate
            clip_epsilon: Clipping epsilon pour PPO
            value_coef: Coefficient de la value loss
            entropy_coef: Coefficient de l'entropy bonus
            max_grad_norm: Max gradient norm pour clipping
            ppo_epochs: Nombre d'époques PPO par update
            use_amp: Utiliser AMP (Mixed Precision)
            target_kl: KL divergence target pour early stopping
            device: Device
        """
        self.policy = policy.to(device)
        
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.use_amp = use_amp
        self.target_kl = target_kl
        self.device = device
        self.initial_lr = learning_rate
        
        # Optimizers
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        
        # AMP Scaler
        self.scaler = GradScaler('cuda', enabled=use_amp)
        
        # Stats
        self.update_count = 0
    
    def update(
        self,
        storage: RolloutStorage,
        mini_batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Effectue une mise à jour PPO.
        
        Args:
            storage: RolloutStorage contenant les rollouts
            mini_batch_size: Taille des mini-batches (None = full batch)
            
        Returns:
            Dict de métriques pour logging
        """
        # Métriques
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        batch_size = storage.num_steps * storage.num_envs
        
        # PPO epochs
        early_stop = False
        for epoch in range(self.ppo_epochs):
            # Générer les mini-batches
            for batch in storage.get_generator(batch_size, mini_batch_size):
                # Extraire les données
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                old_values = batch["values"]
                returns = batch["returns"]
                advantages = batch["advantages"]
                lstm_h = batch["lstm_h"]
                lstm_c = batch["lstm_c"]
                
                # Forward pass avec AMP
                with autocast('cuda', enabled=self.use_amp):
                    # Évaluer les actions
                    new_log_probs, values, entropy = self.policy.evaluate_actions(
                        obs, actions, lstm_state=(lstm_h, lstm_c)
                    )
                    
                    # Si obs est une séquence (3D), flatten les outputs
                    if obs.dim() == 3:
                        # (batch, seq_len) → (batch * seq_len)
                        new_log_probs = new_log_probs.reshape(-1)
                        entropy = entropy.reshape(-1)
                        values = values.reshape(-1)
                    
                    new_values = values if values.dim() == 1 else values.squeeze(-1)
                    
                    # Ratio pour PPO
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Policy loss (clipped)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss (clipped)
                    if self.value_clip_epsilon is not None and self.value_clip_epsilon > 0:
                        v_clipped = old_values + (new_values - old_values).clamp(-self.value_clip_epsilon, self.value_clip_epsilon)
                        v_loss_unclipped = (returns - new_values).pow(2)
                        v_loss_clipped = (returns - v_clipped).pow(2)
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * (returns - new_values).pow(2).mean()
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Métriques
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # KL divergence (approximation)
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    approx_kls.append(approx_kl)
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)
                    
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        early_stop = True
                        break
            if early_stop:
                break
        
        self.update_count += 1
        
        # Retourner les métriques
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "entropy": -np.mean(entropy_losses),  # Positif pour logging
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
            "update_count": self.update_count,
        }
        
        return metrics
    
    
    def set_entropy_coef(self, entropy_coef: float):
        """Met à jour le coefficient d'entropie (pour annealing)."""
        self.entropy_coef = entropy_coef
    
    def set_clip_epsilon(self, clip_epsilon: float):
        """Met à jour le clip epsilon de PPO."""
        self.clip_epsilon = clip_epsilon
    
    def set_learning_rate(self, lr: float):
        """Met à jour le learning rate de l'optimizer."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
    
    def save_checkpoint(self, path: str):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "entropy_coef": self.entropy_coef,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)


class AnnealingScheduler:
    """
    Scheduler pour annealer progressivement un paramètre.
    Utilisé pour entropy_coef, beta_intrinsic, etc.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_steps: int,
        strategy: str = "linear",
    ):
        """
        Args:
            start_value: Valeur initiale
            end_value: Valeur finale
            total_steps: Nombre de steps pour atteindre end_value
            strategy: "linear", "exponential", "cosine"
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.strategy = strategy
        self.current_step = 0
    
    def step(self) -> float:
        """Avance d'un step et retourne la valeur actuelle."""
        if self.current_step >= self.total_steps:
            return self.end_value
        
        progress = self.current_step / self.total_steps
        
        if self.strategy == "linear":
            value = self.start_value + (self.end_value - self.start_value) * progress
        elif self.strategy == "exponential":
            value = self.start_value * (self.end_value / self.start_value) ** progress
        elif self.strategy == "cosine":
            value = self.end_value + (self.start_value - self.end_value) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        else:
            value = self.start_value
        
        self.current_step += 1
        return value
    
    def get_value(self, step: Optional[int] = None) -> float:
        """Retourne la valeur pour un step donné sans avancer le compteur."""
        if step is None:
            step = self.current_step
        
        if step >= self.total_steps:
            return self.end_value
        
        progress = step / self.total_steps
        
        if self.strategy == "linear":
            value = self.start_value + (self.end_value - self.start_value) * progress
        elif self.strategy == "exponential":
            value = self.start_value * (self.end_value / self.start_value) ** progress
        elif self.strategy == "cosine":
            value = self.end_value + (self.start_value - self.end_value) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        else:
            value = self.start_value
        
        return value
