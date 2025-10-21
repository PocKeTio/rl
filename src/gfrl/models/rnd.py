"""
Random Network Distillation (RND) pour curiosity-driven exploration.

Paper: "Exploration by Random Network Distillation" (Burda et al., 2018)
https://arxiv.org/abs/1810.12894

Principe:
- Target network: réseau aléatoire fixe
- Predictor network: apprend à prédire output du target
- Intrinsic reward: erreur de prédiction (MSE)
- États familiers → faible erreur (faible reward)
- États nouveaux → haute erreur (haute reward)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import numpy as np


class RNDNetwork(nn.Module):
    """
    Réseau simple pour RND (target ou predictor).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialisation orthogonale (améliore stabilité)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) ou (batch, seq, obs_dim)
            
        Returns:
            features: (batch, output_dim) ou (batch, seq, output_dim)
        """
        return self.network(obs)


class RND:
    """
    Random Network Distillation pour intrinsic rewards.
    
    Usage:
        rnd = RND(obs_dim=161, device="cuda")
        intrinsic_reward = rnd.compute_intrinsic_reward(obs)
        rnd.update(obs)
    """
    
    def __init__(
        self,
        obs_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        """
        Args:
            obs_dim: Dimension des observations
            output_dim: Dimension de l'embedding RND
            hidden_dim: Taille des couches cachées
            learning_rate: LR pour le predictor
            device: Device
        """
        self.device = device
        
        # Target network (FIXE, jamais entraîné)
        self.target_network = RNDNetwork(obs_dim, output_dim, hidden_dim).to(device)
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Predictor network (entraîné pour prédire target)
        self.predictor_network = RNDNetwork(obs_dim, output_dim, hidden_dim).to(device)
        self.predictor_network.train()
        
        # Optimizer pour predictor
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=learning_rate)
        
        # Running statistics pour normalisation (Welford's algorithm)
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_std = torch.ones(obs_dim, device=device)
        self.obs_count = torch.tensor(1e-4, device=device)  # Éviter division par zéro
        
        # Running statistics pour intrinsic rewards (proper variance tracking)
        self.reward_mean = torch.zeros(1, device=device)
        self.reward_m2 = torch.zeros(1, device=device)  # Sum of squared deviations
        self.reward_count = torch.tensor(0.0, device=device)
        self.reward_std = torch.ones(1, device=device)
    
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise les observations avec running stats."""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def update_obs_stats(self, obs: torch.Tensor):
        """Met à jour les running stats des observations (Welford's algorithm)."""
        with torch.no_grad():
            batch_count = obs.shape[0]
            
            if batch_count == 0:
                return  # Rien à faire
            
            batch_mean = obs.mean(dim=0)
            
            # Welford's online algorithm
            delta = batch_mean - self.obs_mean
            self.obs_count += batch_count
            self.obs_mean += delta * batch_count / self.obs_count
            
            # Update variance (gère batch_count=1 correctement)
            if batch_count == 1:
                # Avec 1 sample: std=0, mais on peut update M2 via delta
                # M2 += delta * delta2 où delta2 = (sample - new_mean)
                delta2 = batch_mean - self.obs_mean
                M2_increment = delta * delta2 * (self.obs_count - 1)
                m_a = self.obs_std.pow(2) * (self.obs_count - 1)
                M2 = m_a + M2_increment
            else:
                # Batch >= 2: calcul standard
                batch_std = obs.std(dim=0, unbiased=False)  # Biased pour Welford
                m_a = self.obs_std.pow(2) * (self.obs_count - batch_count)
                m_b = batch_std.pow(2) * batch_count
                M2 = m_a + m_b + delta.pow(2) * (self.obs_count - batch_count) * batch_count / self.obs_count
            
            self.obs_std = torch.sqrt(M2 / self.obs_count + 1e-8)
    
    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'intrinsic reward (erreur de prédiction).
        
        Args:
            obs: (batch, obs_dim) ou (batch, seq, obs_dim)
            
        Returns:
            intrinsic_reward: (batch,) ou (batch, seq)
        """
        with torch.no_grad():
            # Normaliser obs
            obs_normalized = self.normalize_obs(obs)
            
            # Forward dans les deux réseaux
            target_features = self.target_network(obs_normalized)
            predicted_features = self.predictor_network(obs_normalized)
            
            # MSE comme intrinsic reward (per observation)
            mse = (target_features - predicted_features).pow(2).mean(dim=-1)
            
            # Update running std du MSE avec Welford's algorithm (robust)
            # IMPORTANT: Toujours update, même avec 1 sample (compatibilité num_envs=1)
            if mse.numel() > 0:
                # Welford's algorithm pour variance online
                for mse_val in mse.flatten():
                    self.reward_count += 1
                    delta = mse_val - self.reward_mean
                    self.reward_mean += delta / self.reward_count
                    delta2 = mse_val - self.reward_mean
                    self.reward_m2 += delta * delta2
                
                # Calculer std avec clamp pour éviter collapse
                # Note: reward_count > 1 nécessaire pour variance non-nulle
                if self.reward_count > 1:
                    variance = self.reward_m2 / self.reward_count
                    self.reward_std = torch.sqrt(variance + 1e-8)
                    # CRITICAL: clamp pour éviter explosion/collapse
                    self.reward_std = torch.clamp(self.reward_std, min=0.1, max=10.0)
            
            # Normaliser par std (évite rewards explosifs)
            intrinsic_reward = mse / (self.reward_std + 1e-8)
            
            return intrinsic_reward
    
    def update(self, obs: torch.Tensor) -> float:
        """
        Met à jour le predictor network.
        
        Args:
            obs: (batch, obs_dim)
            
        Returns:
            loss: Prediction loss (float)
        """
        # Update obs stats
        self.update_obs_stats(obs)
        
        # Normaliser obs
        obs_normalized = self.normalize_obs(obs)
        
        # Forward
        with torch.no_grad():
            target_features = self.target_network(obs_normalized)
        
        predicted_features = self.predictor_network(obs_normalized)
        
        # Loss: MSE entre predicted et target
        loss = (predicted_features - target_features).pow(2).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (stabilité)
        torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), 5.0)
        self.optimizer.step()
        
        # Note: reward_std est maintenant updaté dans compute_intrinsic_reward()
        # car on a besoin du std du batch de MSE, pas de la loss moyenne
        
        return loss.item()
    
    def state_dict(self):
        """Sauvegarde pour checkpointing."""
        return {
            'predictor': self.predictor_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'obs_count': self.obs_count,
            'reward_mean': self.reward_mean,
            'reward_m2': self.reward_m2,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }
    
    def load_state_dict(self, state_dict):
        """Chargement depuis checkpoint (avec backward compatibility)."""
        self.predictor_network.load_state_dict(state_dict['predictor'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.obs_mean = state_dict['obs_mean']
        self.obs_std = state_dict['obs_std']
        self.obs_count = state_dict['obs_count']
        
        # Backward compatibility: anciens checkpoints n'ont pas reward_mean/m2
        if 'reward_mean' in state_dict:
            self.reward_mean = state_dict['reward_mean']
        if 'reward_m2' in state_dict:
            self.reward_m2 = state_dict['reward_m2']
        
        self.reward_std = state_dict['reward_std']
        self.reward_count = state_dict['reward_count']
