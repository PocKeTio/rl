"""
Policy réseau avec LSTM pour Google Research Football.

Architecture:
1. Backbone: Linear(obs_dim → 512) → ReLU → LayerNorm → Linear(512 → 512) → ReLU
2. LSTM: LSTM(512 → 128)
3. Heads: 
   - Policy: Linear(128 → n_actions)
   - Value: Linear(128 → 1)

Initialisation orthogonale pour stabilité.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PolicyLSTM(nn.Module):
    """
    Policy réseau avec LSTM et heads policy/value.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 512,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
    ):
        """
        Args:
            obs_dim: Dimension de l'observation
            action_dim: Nombre d'actions possibles
            hidden_size: Taille des couches cachées du backbone
            lstm_hidden_size: Taille du hidden state LSTM
            lstm_num_layers: Nombre de couches LSTM
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # Backbone MLP
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        
        # Heads
        self.policy_head = nn.Linear(lstm_hidden_size, action_dim)
        self.value_head = nn.Linear(lstm_hidden_size, 1)
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation orthogonale des poids."""
        # Backbone
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # LSTM (tanh pour stabilité)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        
        # Heads
        nn.init.orthogonal_(self.policy_head.weight, gain=0.5)  # Augmenté pour forcer apprentissage
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: Observations (batch_size, seq_len, obs_dim) ou (batch_size, obs_dim)
            lstm_state: Tuple (h, c) de taille (num_layers, batch_size, lstm_hidden)
            done: Masque des épisodes terminés (batch_size,) pour reset LSTM state
            
        Returns:
            Tuple (logits, values, new_lstm_state)
            - logits: (batch_size, action_dim) ou (batch_size, seq_len, action_dim)
            - values: (batch_size, 1) ou (batch_size, seq_len, 1)
            - new_lstm_state: Tuple (h, c)
        """
        # Déterminer si on a une séquence
        is_sequence = obs.dim() == 3
        if not is_sequence:
            obs = obs.unsqueeze(1)  # (batch, obs_dim) → (batch, 1, obs_dim)
        
        batch_size, seq_len, _ = obs.shape
        
        # Backbone
        features = self.backbone(obs)  # (batch, seq_len, hidden_size)
        
        # LSTM
        if lstm_state is None:
            # Init state
            h = torch.zeros(
                self.lstm_num_layers, batch_size, self.lstm_hidden_size,
                device=obs.device, dtype=obs.dtype
            )
            c = torch.zeros(
                self.lstm_num_layers, batch_size, self.lstm_hidden_size,
                device=obs.device, dtype=obs.dtype
            )
            lstm_state = (h, c)
        
        # Reset LSTM state pour les épisodes terminés
        if done is not None:
            h, c = lstm_state
            # done: (batch,) → (1, batch, 1)
            done_mask = done.view(1, batch_size, 1).float()
            h = h * (1.0 - done_mask)
            c = c * (1.0 - done_mask)
            lstm_state = (h, c)
        
        lstm_out, new_lstm_state = self.lstm(features, lstm_state)
        # lstm_out: (batch, seq_len, lstm_hidden)
        
        # Heads
        logits = self.policy_head(lstm_out)  # (batch, seq_len, action_dim)
        values = self.value_head(lstm_out)  # (batch, seq_len, 1)
        
        # Si on n'avait pas de séquence, remove seq dimension
        if not is_sequence:
            logits = logits.squeeze(1)  # (batch, action_dim)
            values = values.squeeze(1)  # (batch, 1)
        
        return logits, values, new_lstm_state
    
    def get_action(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Échantillonne une action depuis la policy.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            lstm_state: État LSTM
            deterministic: Si True, prend l'action avec la plus grande probabilité
            
        Returns:
            Tuple (action, log_prob, value, new_lstm_state)
        """
        logits, values, new_lstm_state = self.forward(obs, lstm_state)
        
        # Distribution catégorielle
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, values, new_lstm_state
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Évalue des actions données (pour PPO update).
        
        Args:
            obs: Observations (batch_size, obs_dim) ou (batch_size, seq_len, obs_dim)
            actions: Actions (batch_size,) ou (batch_size, seq_len)
            lstm_state: État LSTM
            done: Masque done
            
        Returns:
            Tuple (log_probs, values, entropy)
        """
        logits, values, _ = self.forward(obs, lstm_state, done)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy
    
    def get_value(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Calcule uniquement la valeur (sans sampling d'action).
        
        Args:
            obs: Observations (batch_size, obs_dim)
            lstm_state: État LSTM
            
        Returns:
            values: (batch_size, 1)
        """
        _, values, _ = self.forward(obs, lstm_state)
        return values
