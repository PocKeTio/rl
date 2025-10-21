"""
Tracker de statistiques de football pour TensorBoard.
Métriques détaillées comme un vrai match de foot.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import deque


class FootballStatsTracker:
    """
    Tracker de statistiques de football avancées.
    """
    
    def __init__(self, num_envs: int, window_size: int = 100):
        """
        Args:
            num_envs: Nombre d'environnements parallèles
            window_size: Taille de la fenêtre pour moyennes glissantes
        """
        self.num_envs = num_envs
        self.window_size = window_size
        
        # Stats par épisode (moyennes glissantes)
        self.possession_pct = deque(maxlen=window_size)
        self.shots_per_game = deque(maxlen=window_size)
        self.shots_on_target_per_game = deque(maxlen=window_size)
        self.passes_per_game = deque(maxlen=window_size)
        self.pass_accuracy = deque(maxlen=window_size)
        self.distance_traveled = deque(maxlen=window_size)
        
        # Zones de jeu (% du temps dans chaque zone)
        self.time_in_defense = deque(maxlen=window_size)
        self.time_in_midfield = deque(maxlen=window_size)
        self.time_in_attack = deque(maxlen=window_size)
        
        # Heatmap (positions cumulées)
        self.heatmap_resolution = (20, 14)  # Résolution de la heatmap
        self.player_heatmap = np.zeros(self.heatmap_resolution)
        self.ball_heatmap = np.zeros(self.heatmap_resolution)
        
        # Tracking par step (pour calculer stats d'épisode)
        self.episode_stats = [self._init_episode_stats() for _ in range(num_envs)]
        
    def _init_episode_stats(self) -> Dict[str, Any]:
        """Initialise les stats pour un épisode."""
        return {
            'steps_with_ball': 0,
            'total_steps': 0,
            'shots': 0,
            'shots_on_target': 0,
            'passes_attempted': 0,
            'passes_completed': 0,
            'distance': 0.0,
            'steps_in_defense': 0,
            'steps_in_midfield': 0,
            'steps_in_attack': 0,
            'prev_player_pos': None,
        }
    
    def update_step(self, env_idx: int, obs: Dict[str, Any], action: int, info: Dict[str, Any]):
        """
        Met à jour les stats pour un step.
        
        Args:
            env_idx: Index de l'environnement
            obs: Observation (raw dict)
            action: Action prise
            info: Info dict
        """
        stats = self.episode_stats[env_idx]
        stats['total_steps'] += 1
        
        # Possession
        ball_owned_team = obs.get('ball_owned_team', -1)
        if ball_owned_team == 0:  # Notre équipe
            stats['steps_with_ball'] += 1
        
        # Position du joueur actif
        active_idx = obs.get('active', 0)
        left_team = obs.get('left_team', [])
        if active_idx < len(left_team):
            player_pos = left_team[active_idx]
            if isinstance(player_pos, (list, np.ndarray)) and len(player_pos) >= 2:
                x, y = player_pos[0], player_pos[1]
                
                # Zone de jeu (x: -1 à 1)
                if x < -0.33:
                    stats['steps_in_defense'] += 1
                elif x < 0.33:
                    stats['steps_in_midfield'] += 1
                else:
                    stats['steps_in_attack'] += 1
                
                # Distance parcourue
                if stats['prev_player_pos'] is not None:
                    prev_x, prev_y = stats['prev_player_pos']
                    dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    stats['distance'] += dist
                
                stats['prev_player_pos'] = (x, y)
                
                # Heatmap joueur
                self._add_to_heatmap(self.player_heatmap, x, y)
        
        # Position du ballon
        ball = obs.get('ball', [0, 0, 0])
        if isinstance(ball, (list, np.ndarray)) and len(ball) >= 2:
            ball_x, ball_y = ball[0], ball[1]
            self._add_to_heatmap(self.ball_heatmap, ball_x, ball_y)
        
        # Tirs (action 12 = SHOT)
        if action == 12:
            stats['shots'] += 1
            
            # Tir cadré si direction vers le but
            ball_direction = obs.get('ball_direction', [0, 0, 0])
            if isinstance(ball_direction, (list, np.ndarray)) and len(ball_direction) >= 1:
                if ball_direction[0] > 0:  # Vers le but adverse (x > 0)
                    stats['shots_on_target'] += 1
        
        # Passes (détection via changement de ball_owned_player)
        # Note: Cette détection est approximative
        if 'prev_ball_owner' in stats:
            prev_owner = stats['prev_ball_owner']
            curr_owner = obs.get('ball_owned_player', -1)
            curr_team = obs.get('ball_owned_team', -1)
            
            if curr_team == 0 and prev_owner != curr_owner and prev_owner != -1 and curr_owner != -1:
                stats['passes_attempted'] += 1
                stats['passes_completed'] += 1
        
        stats['prev_ball_owner'] = obs.get('ball_owned_player', -1)
    
    def _add_to_heatmap(self, heatmap: np.ndarray, x: float, y: float):
        """
        Ajoute une position à la heatmap.
        
        Args:
            heatmap: Heatmap à mettre à jour
            x: Position x (terrain: -1 à 1)
            y: Position y (terrain: -0.42 à 0.42)
        """
        # Convertir coordonnées terrain en indices heatmap
        # x: -1 à 1 → 0 à width-1
        # y: -0.42 à 0.42 → 0 à height-1
        width, height = self.heatmap_resolution
        
        i = int((x + 1.0) / 2.0 * (width - 1))
        j = int((y + 0.42) / 0.84 * (height - 1))
        
        # Clamp
        i = max(0, min(width - 1, i))
        j = max(0, min(height - 1, j))
        
        heatmap[i, j] += 1
    
    def on_episode_done(self, env_idx: int) -> Dict[str, float]:
        """
        Appelé quand un épisode se termine.
        Calcule et retourne les stats de l'épisode.
        
        Args:
            env_idx: Index de l'environnement
            
        Returns:
            Dict de métriques pour cet épisode
        """
        stats = self.episode_stats[env_idx]
        
        # Calculer les métriques
        total_steps = max(stats['total_steps'], 1)
        
        possession = (stats['steps_with_ball'] / total_steps) * 100
        shots = stats['shots']
        shots_on_target = stats['shots_on_target']
        passes_attempted = stats['passes_attempted']
        passes_completed = stats['passes_completed']
        pass_acc = (passes_completed / max(passes_attempted, 1)) * 100
        distance = stats['distance']
        
        time_defense = (stats['steps_in_defense'] / total_steps) * 100
        time_midfield = (stats['steps_in_midfield'] / total_steps) * 100
        time_attack = (stats['steps_in_attack'] / total_steps) * 100
        
        # Ajouter aux deques
        self.possession_pct.append(possession)
        self.shots_per_game.append(shots)
        self.shots_on_target_per_game.append(shots_on_target)
        self.passes_per_game.append(passes_attempted)
        self.pass_accuracy.append(pass_acc)
        self.distance_traveled.append(distance)
        self.time_in_defense.append(time_defense)
        self.time_in_midfield.append(time_midfield)
        self.time_in_attack.append(time_attack)
        
        # Reset stats pour cet env
        self.episode_stats[env_idx] = self._init_episode_stats()
        
        return {
            'possession_pct': possession,
            'shots': shots,
            'shots_on_target': shots_on_target,
            'passes_attempted': passes_attempted,
            'pass_accuracy': pass_acc,
            'distance_traveled': distance,
            'time_in_defense': time_defense,
            'time_in_midfield': time_midfield,
            'time_in_attack': time_attack,
        }
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Retourne les stats moyennes sur la fenêtre.
        
        Returns:
            Dict de métriques moyennes
        """
        if len(self.possession_pct) == 0:
            return {}
        
        return {
            'possession_pct': np.mean(self.possession_pct),
            'shots_per_game': np.mean(self.shots_per_game),
            'shots_on_target_per_game': np.mean(self.shots_on_target_per_game),
            'shot_accuracy': (np.mean(self.shots_on_target_per_game) / max(np.mean(self.shots_per_game), 0.01)) * 100,
            'passes_per_game': np.mean(self.passes_per_game),
            'pass_accuracy': np.mean(self.pass_accuracy),
            'distance_traveled': np.mean(self.distance_traveled),
            'time_in_defense': np.mean(self.time_in_defense),
            'time_in_midfield': np.mean(self.time_in_midfield),
            'time_in_attack': np.mean(self.time_in_attack),
        }
    
    def get_heatmaps(self) -> Dict[str, np.ndarray]:
        """
        Retourne les heatmaps normalisées.
        
        Returns:
            Dict avec 'player' et 'ball' heatmaps
        """
        # Normaliser
        player_hm = self.player_heatmap / (self.player_heatmap.max() + 1e-8)
        ball_hm = self.ball_heatmap / (self.ball_heatmap.max() + 1e-8)
        
        return {
            'player': player_hm,
            'ball': ball_hm,
        }
    
    def reset_heatmaps(self):
        """Reset les heatmaps (appelé périodiquement pour éviter saturation)."""
        self.player_heatmap = np.zeros(self.heatmap_resolution)
        self.ball_heatmap = np.zeros(self.heatmap_resolution)
