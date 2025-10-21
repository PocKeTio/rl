"""
Encodeur d'observations pour Google Research Football (représentation 'raw').

L'encodeur transforme le dictionnaire d'observations raw en un vecteur 1D utilisable par les réseaux neuronaux.

Structure de l'observation encodée:
1. Ballon (6 dim): position [x,y,z] + vélocité [vx,vy,vz]
2. Joueurs gauche (11 × 5 = 55 dim): [x, y, direction_x, direction_y, vitesse] × 11
3. Joueurs droite (11 × 5 = 55 dim): idem
4. Possession:
   - ball_owned_team: one-hot 3 dim (-1, 0, 1)
   - ball_owned_player: one-hot 11 dim (0-10)
   - owned_by_me: 1 dim (booléen)
5. Joueur actif: one-hot 11 dim (0-10)
6. Sticky actions: 10 dim (booléens)
7. Contexte:
   - steps_left: 1 dim (normalisé)
   - score: 1 dim (différence de buts normalisée)
   - game_mode: one-hot 7 dim

Total: 6 + 55 + 55 + 3 + 11 + 1 + 11 + 10 + 1 + 1 + 7 = 161 dimensions
"""

import numpy as np
from typing import Dict, Any, Tuple


class RawObsEncoder:
    """
    Encode les observations raw de Google Research Football en vecteur dense.
    
    Références:
    - https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
    - https://di-engine-docs.readthedocs.io/
    """
    
    def __init__(
        self,
        include_score: bool = True,
        normalize_positions: bool = True,
        max_steps: int = 3001,  # Steps max dans un match GRF
    ):
        """
        Args:
            include_score: Inclure la différence de buts dans l'observation
            normalize_positions: Normaliser positions et vitesses
            max_steps: Nombre max de steps pour normaliser steps_left
        """
        self.include_score = include_score
        self.normalize_positions = normalize_positions
        self.max_steps = max_steps
        
        # Dimensions calculées
        self._obs_dim = self._calculate_obs_dim()
    
    def _calculate_obs_dim(self) -> int:
        """Calcule la dimension totale de l'observation encodée."""
        dim = 0
        dim += 6  # Ballon (x,y,z, vx,vy,vz)
        dim += 11 * 5  # Joueurs gauche (x,y,dx,dy,speed) × 11
        dim += 11 * 5  # Joueurs droite
        dim += 3  # ball_owned_team (one-hot)
        dim += 11  # ball_owned_player (one-hot)
        dim += 1  # owned_by_me
        dim += 11  # active player (one-hot)
        dim += 10  # sticky_actions
        dim += 1  # steps_left (normalized)
        if self.include_score:
            dim += 1  # score difference
        dim += 7  # game_mode (one-hot)
        return dim
    
    @property
    def obs_dim(self) -> int:
        """Dimension de l'observation encodée."""
        return self._obs_dim
    
    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Encode une observation raw en vecteur.
        
        Args:
            obs: Dictionnaire d'observation raw de GRF
            
        Returns:
            Vecteur 1D de dimension self.obs_dim
        """
        features = []
        
        # 1. Ballon (6 dim)
        ball = obs["ball"]  # [x, y, z]
        ball_direction = obs["ball_direction"]  # [vx, vy, vz]
        if self.normalize_positions:
            ball = np.clip(ball, -1.0, 1.0)
            ball_direction = np.clip(ball_direction, -1.0, 1.0)
        features.extend(ball)
        features.extend(ball_direction)
        
        # 2. Joueurs gauche (11 × 5 = 55 dim, paddé si nécessaire)
        left_team = obs["left_team"]  # (n, 2) positions (n peut être < 11)
        left_team_direction = obs["left_team_direction"]  # (n, 2) directions
        num_left = left_team.shape[0]
        
        for i in range(11):
            if i < num_left:
                # Joueur existe
                pos = left_team[i]  # [x, y]
                direction = left_team_direction[i]  # [dx, dy]
                speed = np.linalg.norm(direction)
                
                if self.normalize_positions:
                    pos = np.clip(pos, -1.0, 1.0)
                    direction = np.clip(direction, -1.0, 1.0)
                    speed = min(speed, 1.0)
                
                features.extend([pos[0], pos[1], direction[0], direction[1], speed])
            else:
                # Joueur absent -> padding avec zéros
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 3. Joueurs droite (11 × 5 = 55 dim, paddé si nécessaire)
        right_team = obs["right_team"]
        right_team_direction = obs["right_team_direction"]
        num_right = right_team.shape[0]
        
        for i in range(11):
            if i < num_right:
                # Joueur existe
                pos = right_team[i]
                direction = right_team_direction[i]
                speed = np.linalg.norm(direction)
                
                if self.normalize_positions:
                    pos = np.clip(pos, -1.0, 1.0)
                    direction = np.clip(direction, -1.0, 1.0)
                    speed = min(speed, 1.0)
                
                features.extend([pos[0], pos[1], direction[0], direction[1], speed])
            else:
                # Joueur absent -> padding avec zéros
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 4. Possession
        # ball_owned_team: -1 (personne), 0 (gauche/nous), 1 (droite/adversaire)
        ball_owned_team = obs["ball_owned_team"]
        ball_owned_team_onehot = self._to_onehot(ball_owned_team + 1, 3)  # Shift to 0-2
        features.extend(ball_owned_team_onehot)
        
        # ball_owned_player: 0-10 (index du joueur qui possède la balle)
        # -1 si personne ne possède
        ball_owned_player = obs["ball_owned_player"]
        if ball_owned_player == -1:
            ball_owned_player_onehot = np.zeros(11)
        else:
            ball_owned_player_onehot = self._to_onehot(ball_owned_player, 11)
        features.extend(ball_owned_player_onehot)
        
        # owned_by_me: booléen indiquant si notre équipe possède la balle
        # Calculé à partir de ball_owned_team
        owned_by_me = 1.0 if ball_owned_team == 0 else 0.0
        features.append(owned_by_me)
        
        # 5. Joueur actif (one-hot 11 dim)
        active = obs["active"]  # Index 0-10
        active_onehot = self._to_onehot(active, 11)
        features.extend(active_onehot)
        
        # 6. Sticky actions (10 booléens)
        sticky_actions = obs["sticky_actions"]  # Liste de 10 booléens
        # sticky[0:8] = directions (top, top-right, ..., top-left)
        # sticky[8] = sprint
        # sticky[9] = dribble
        features.extend([float(x) for x in sticky_actions])
        
        # 7. Contexte
        # steps_left: nombre de steps restants (normalisé)
        steps_left = obs["steps_left"]
        steps_left_norm = steps_left / self.max_steps
        features.append(steps_left_norm)
        
        # score: différence de buts (optionnel)
        if self.include_score:
            score = obs["score"]  # [score_left, score_right]
            score_diff = (score[0] - score[1]) / 10.0  # Normalisation approximative
            score_diff = np.clip(score_diff, -1.0, 1.0)
            features.append(score_diff)
        
        # game_mode: mode de jeu (one-hot 7 dim)
        # 0: Normal, 1: KickOff, 2: GoalKick, 3: FreeKick, 4: Corner, 5: ThrowIn, 6: Penalty
        game_mode = obs["game_mode"]
        game_mode_onehot = self._to_onehot(game_mode, 7)
        features.extend(game_mode_onehot)
        
        # Conversion en numpy array
        encoded_obs = np.array(features, dtype=np.float32)
        
        # Vérification de la dimension
        assert encoded_obs.shape[0] == self.obs_dim, \
            f"Dimension mismatch: expected {self.obs_dim}, got {encoded_obs.shape[0]}"
        
        return encoded_obs
    
    def encode_batch(self, obs_batch: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Encode un batch d'observations.
        
        Args:
            obs_batch: Dict avec clés identiques à obs mais valeurs de shape (batch_size, ...)
            
        Returns:
            Tableau de shape (batch_size, obs_dim)
        """
        # Extraire batch_size
        batch_size = obs_batch["ball"].shape[0]
        
        # Encoder chaque observation individuellement
        encoded_batch = []
        for i in range(batch_size):
            obs_i = {k: v[i] for k, v in obs_batch.items()}
            encoded_obs = self.encode(obs_i)
            encoded_batch.append(encoded_obs)
        
        return np.stack(encoded_batch, axis=0)
    
    @staticmethod
    def _to_onehot(index: int, size: int) -> np.ndarray:
        """Convertit un index en one-hot encoding."""
        onehot = np.zeros(size, dtype=np.float32)
        if 0 <= index < size:
            onehot[index] = 1.0
        return onehot
    
    def get_obs_info(self) -> Dict[str, Tuple[int, int]]:
        """
        Retourne les ranges d'indices pour chaque composante de l'observation.
        Utile pour le debugging et l'analyse.
        
        Returns:
            Dict: {nom_composante: (start_idx, end_idx)}
        """
        ranges = {}
        idx = 0
        
        # Ballon
        ranges["ball"] = (idx, idx + 6)
        idx += 6
        
        # Joueurs
        ranges["left_team"] = (idx, idx + 55)
        idx += 55
        ranges["right_team"] = (idx, idx + 55)
        idx += 55
        
        # Possession
        ranges["ball_owned_team"] = (idx, idx + 3)
        idx += 3
        ranges["ball_owned_player"] = (idx, idx + 11)
        idx += 11
        ranges["owned_by_me"] = (idx, idx + 1)
        idx += 1
        
        # Actif
        ranges["active"] = (idx, idx + 11)
        idx += 11
        
        # Sticky
        ranges["sticky_actions"] = (idx, idx + 10)
        idx += 10
        
        # Contexte
        ranges["steps_left"] = (idx, idx + 1)
        idx += 1
        if self.include_score:
            ranges["score"] = (idx, idx + 1)
            idx += 1
        ranges["game_mode"] = (idx, idx + 7)
        idx += 7
        
        return ranges
    
    def decode_component(self, encoded_obs: np.ndarray, component: str) -> np.ndarray:
        """
        Extrait une composante spécifique de l'observation encodée.
        
        Args:
            encoded_obs: Observation encodée
            component: Nom de la composante (ex: "ball", "active", "sticky_actions")
            
        Returns:
            Sous-vecteur correspondant à la composante
        """
        ranges = self.get_obs_info()
        if component not in ranges:
            raise ValueError(f"Unknown component: {component}. Available: {list(ranges.keys())}")
        
        start, end = ranges[component]
        return encoded_obs[start:end]
