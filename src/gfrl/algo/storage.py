"""
Rollout Storage LSTM-aware pour PPO.

Stocke les trajectoires d'expérience collectées pendant les rollouts:
- Observations
- Actions
- Log probs
- Rewards
- Values
- Dones
- États LSTM
- Infos
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


class RolloutStorage:
    """
    Buffer pour stocker les rollouts avec support LSTM.
    """
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_space,
        lstm_hidden_size: int,
        lstm_num_layers: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            num_steps: Nombre de steps par rollout
            num_envs: Nombre d'environnements parallèles
            obs_shape: Shape des observations
            action_space: Action space (pour déterminer le type d'action)
            lstm_hidden_size: Taille du hidden state LSTM
            lstm_num_layers: Nombre de couches LSTM
            device: Device ('cpu' ou 'cuda')
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        
        # Buffers
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps + 1, num_envs, device=device)
        self.dones = torch.zeros(num_steps + 1, num_envs, dtype=torch.bool, device=device)
        
        # LSTM states: (num_steps + 1, num_layers, num_envs, hidden_size)
        self.lstm_h = torch.zeros(
            num_steps + 1, lstm_num_layers, num_envs, lstm_hidden_size, device=device
        )
        self.lstm_c = torch.zeros(
            num_steps + 1, lstm_num_layers, num_envs, lstm_hidden_size, device=device
        )
        
        # Intrinsic rewards (optionnel, pour RND)
        self.intrinsic_rewards = torch.zeros(num_steps, num_envs, device=device)
        
        # Advantages et returns (calculés par compute_returns)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        
        # Step counter
        self.step = 0
    
    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        intrinsic_reward: Optional[torch.Tensor] = None,
    ):
        """
        Insère une transition dans le buffer.
        
        Args:
            obs: Observations (num_envs, *obs_shape)
            action: Actions (num_envs,)
            log_prob: Log probs (num_envs,)
            value: Values (num_envs,)
            reward: Rewards (num_envs,)
            done: Dones (num_envs,)
            lstm_state: Tuple (h, c) de shape (num_layers, num_envs, hidden_size)
            intrinsic_reward: Récompenses intrinsèques optionnelles (num_envs,)
        """
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        
        # S'assurer que value a la shape correcte [num_envs]
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(value) if hasattr(value, '__array__') else torch.tensor(value)
        
        if value.dim() == 0:
            # Scalaire -> [1]
            value = value.unsqueeze(0)
        elif value.dim() > 1 and value.shape[-1] == 1:
            # [num_envs, 1] -> [num_envs]
            value = value.squeeze(-1)
        self.values[self.step].copy_(value)
        
        # S'assurer que reward a la shape correcte [num_envs]
        if reward.dim() == 0:
            # Scalaire -> [1]
            reward = reward.unsqueeze(0)
        elif reward.dim() > 1:
            # [num_envs, 1] -> [num_envs]
            reward = reward.squeeze(-1)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step + 1].copy_(done)
        
        # LSTM state
        h, c = lstm_state
        self.lstm_h[self.step + 1].copy_(h)
        self.lstm_c[self.step + 1].copy_(c)
        
        if intrinsic_reward is not None:
            self.intrinsic_rewards[self.step].copy_(intrinsic_reward)
        
        self.step += 1
    
    def after_update(self):
        """
        Copie la dernière observation et LSTM state au début du buffer.
        Appelé après chaque update PPO.
        """
        self.obs[0].copy_(self.obs[-1])
        self.dones[0].copy_(self.dones[-1])
        self.lstm_h[0].copy_(self.lstm_h[-1])
        self.lstm_c[0].copy_(self.lstm_c[-1])
        self.step = 0
    
    def compute_returns(
        self,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_intrinsic: bool = False,
        beta_intrinsic: float = 0.0,
    ):
        """
        Calcule les returns et advantages avec GAE (Generalized Advantage Estimation).
        
        Args:
            next_value: Valeur du dernier état (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda
            use_intrinsic: Utiliser les récompenses intrinsèques
            beta_intrinsic: Poids des récompenses intrinsèques
        """
        self.values[-1] = next_value
        gae = 0
        
        for step in reversed(range(self.num_steps)):
            # Récompense totale
            reward = self.rewards[step]
            if use_intrinsic and beta_intrinsic > 0:
                reward = reward + beta_intrinsic * self.intrinsic_rewards[step]
            
            # Masque pour les épisodes non terminés
            mask = (~self.dones[step + 1]).float()
            
            # TD error
            delta = reward + gamma * self.values[step + 1] * mask - self.values[step]
            
            # GAE
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
    
    def get_generator(
        self,
        batch_size: int,
        mini_batch_size: Optional[int] = None,
    ):
        """
        Générateur de mini-batches pour l'entraînement PPO.
        
        Args:
            batch_size: Taille totale du batch (num_steps * num_envs)
            mini_batch_size: Taille des mini-batches (None = full batch)
            
        Yields:
            Dict contenant les tenseurs pour un mini-batch
        """
        # Flatten (num_steps, num_envs, ...) -> (batch_size, ...)
        obs_batch = self.obs[:-1].reshape(-1, *self.obs_shape)
        actions_batch = self.actions.reshape(-1)
        log_probs_batch = self.log_probs.reshape(-1)
        values_batch = self.values[:-1].reshape(-1)
        returns_batch = self.returns.reshape(-1)
        advantages_batch = self.advantages.reshape(-1)
        dones_batch = self.dones[:-1].reshape(-1)
        
        # LSTM states au début de chaque séquence
        # On prend les états au début de chaque rollout (step 0)
        lstm_h_batch = self.lstm_h[0]  # (num_layers, num_envs, hidden_size)
        lstm_c_batch = self.lstm_c[0]
        
        # Normaliser les advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )
        
        if mini_batch_size is None:
            # Full batch - reshape en séquences pour LSTM
            # (num_steps, num_envs, ...) → (num_envs, num_steps, ...)
            obs_seq = self.obs[:-1].permute(1, 0, 2)  # (num_envs, num_steps, obs_dim)
            actions_seq = self.actions.permute(1, 0)  # (num_envs, num_steps)
            
            # Reshape autres tensors pour match
            log_probs_seq = self.log_probs.permute(1, 0).reshape(-1)  # (num_envs * num_steps)
            values_seq = self.values[:-1].permute(1, 0).reshape(-1)
            returns_seq = self.returns.permute(1, 0).reshape(-1)
            advantages_seq = self.advantages.permute(1, 0).reshape(-1)
            
            # Normaliser advantages après reshape
            advantages_seq = (advantages_seq - advantages_seq.mean()) / (advantages_seq.std() + 1e-8)
            
            yield {
                "obs": obs_seq,  # (num_envs, num_steps, obs_dim)
                "actions": actions_seq,  # (num_envs, num_steps)
                "log_probs": log_probs_seq,  # (num_envs * num_steps)
                "values": values_seq,
                "returns": returns_seq,
                "advantages": advantages_seq,
                "dones": None,  # Pas utilisé en full batch
                "lstm_h": lstm_h_batch,  # (num_layers, num_envs, hidden_size)
                "lstm_c": lstm_c_batch,
            }
        else:
            # Mini-batches
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                if end > batch_size:
                    end = batch_size
                
                mb_indices = indices[start:end]
                mb_size = len(mb_indices)
                
                # Créer un LSTM state avec la bonne taille de batch
                # Pour simplifier, on utilise un state zéro pour chaque mini-batch
                mb_lstm_h = torch.zeros(
                    self.lstm_num_layers, mb_size, self.lstm_hidden_size, device=self.device
                )
                mb_lstm_c = torch.zeros(
                    self.lstm_num_layers, mb_size, self.lstm_hidden_size, device=self.device
                )
                
                yield {
                    "obs": obs_batch[mb_indices],
                    "actions": actions_batch[mb_indices],
                    "log_probs": log_probs_batch[mb_indices],
                    "values": values_batch[mb_indices],
                    "returns": returns_batch[mb_indices],
                    "advantages": advantages_batch[mb_indices],
                    "dones": dones_batch[mb_indices],
                    "lstm_h": mb_lstm_h,
                    "lstm_c": mb_lstm_c,
                }
    
    def get_stats(self) -> Dict[str, float]:
        """Retourne des statistiques pour logging."""
        return {
            "mean_reward": self.rewards.mean().item(),
            "mean_value": self.values[:-1].mean().item(),
            "mean_advantage": self.advantages.mean().item(),
            "mean_intrinsic_reward": self.intrinsic_rewards.mean().item(),
        }
