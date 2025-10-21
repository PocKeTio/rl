"""
Trainer principal avec curriculum learning et self-play gelé.

Fonctionnalités:
- Boucle d'entraînement PPO
- Curriculum learning (progression entre scénarios)
- Self-play gelé (vs checkpoints passés)
- Opponent sampling (easy/default/hard)
- Annealing (entropy, shaping)
- Logging (TensorBoard/W&B)
- Checkpointing
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from collections import deque
import time

from ..models.policy_lstm import PolicyLSTM
from ..models.rnd import RND
from ..algo.ppo import PPO
from ..algo.storage import RolloutStorage
from ..env.make_env import create_vec_envs
from ..env.rewarders import RewardShaper
from ..utils.logging import setup_logger
from ..utils.schedulers import AnnealingScheduler
from ..utils.football_stats import FootballStatsTracker

logger = setup_logger("gfrl.trainer")


class Trainer:
    """
    Trainer principal pour PPO+LSTM avec curriculum.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config_path: Chemin vers le fichier de config principal
            config_dict: Ou directement un dict de config
        """
        # Charger la config
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Device
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Curriculum - Charger AVANT de créer les envs
        self._load_curriculum()
        self.current_phase_idx = 0
        self.phase_start_step = 0
        
        # Si curriculum actif, utiliser le premier env de la première phase
        if self.curriculum_phases and len(self.curriculum_phases) > 0:
            first_phase_env = self.curriculum_phases[0].get("env_name")
            if first_phase_env:
                self.config['env_name'] = first_phase_env
                logger.info(f"Curriculum active: Starting with {first_phase_env}")
        
        # Créer les composants (maintenant que l'env_name est configuré)
        self._create_env()
        self._create_models()
        self._create_ppo()
        self._create_storage()
        self._create_schedulers()
        
        # Self-play gelé
        self.checkpoint_pool: List[Path] = []
        
        # Stats
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths_deque = deque(maxlen=100)  # Renamed to avoid conflict
        self.best_mean_reward = -np.inf
        
        # Stats tracking simple (sans modules externes)
        self.metrics_tracker = None
        self.episode_trackers = None
        
        # Stats manuelles pour logging
        self.total_goals_scored = 0
        self.total_goals_conceded = 0
        self.total_own_goals = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.total_episodes_completed = 0
        
        # Tracking manuel des épisodes (par env)
        num_envs = self.config.get("num_envs", 48)
        self.episode_returns = np.zeros(num_envs)
        self.episode_lengths = np.zeros(num_envs, dtype=int)
        
        # Évaluation intermédiaire désactivée (module supprimé)
        self.evaluator = None
        
        # Position tracking désactivé (modules supprimés)
        self.position_tracker = None
        self.ball_tracker = None
        
        # Football stats tracker
        self.football_stats = FootballStatsTracker(num_envs=num_envs, window_size=100)
        
        # Logging
        self._setup_logging()
    
    def _create_env(self):
        """Crée les environnements vectorisés."""
        num_envs = self.config.get("num_envs", 48)
        env_name = self.config.get("env_name", "11_vs_11_stochastic")
        
        # LOG CRITIQUE : Vérifier quel env est créé
        logger.warning(f"🔍 CREATING ENVS: env_name='{env_name}', num_envs={num_envs}")
        
        # Reward shaper
        reward_config = self.config.get("reward_config", "configs/rewards.yaml")
        self.reward_shaper = RewardShaper(
            config_path=Path(reward_config)
        )
        
        # Charger running_norm depuis config
        running_norm = self.config.get("running_norm", False)
        
        self.envs = create_vec_envs(
            num_envs=num_envs,
            env_name=env_name,
            representation="raw",
            action_set="sticky",
            frame_skip=self.config.get("frame_skip", 3),
            obs_wrapper=True,
            running_norm=running_norm,
            mirror=self.config.get("mirror", False),
            reward_shaper=self.reward_shaper,
        )
        
        # Extraire les dimensions
        self.obs_dim = self.envs.single_observation_space.shape[0]
        self.action_dim = self.envs.single_action_space.n
        
        logger.info(f"Created {num_envs} vectorized environments")
        logger.info(f"  Environment: {env_name}")
        logger.info(f"  Observation dim: {self.obs_dim}")
        logger.info(f"  Action dim: {self.action_dim}")
    
    def _create_models(self):
        """Crée le modèle policy LSTM + RND (si activé)."""
        # Policy LSTM
        self.policy = PolicyLSTM(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=self.config.get("hidden_size", 512),
            lstm_hidden_size=self.config.get("lstm_hidden_size", 128),
            lstm_num_layers=self.config.get("lstm_num_layers", 1),
        )
        
        # RND (curiosity-driven exploration)
        self.use_rnd = self.config.get("use_rnd", False)
        if self.use_rnd:
            self.rnd = RND(
                obs_dim=self.obs_dim,
                output_dim=self.config.get("rnd_output_dim", 128),
                hidden_dim=self.config.get("rnd_hidden_dim", 256),
                learning_rate=self.config.get("rnd_lr", 1e-4),
                device=self.device,
            )
            logger.info("Created RND module for curiosity")
            logger.info(f"  RND parameters: {sum(p.numel() for p in self.rnd.predictor_network.parameters()):,}")
        else:
            self.rnd = None
        
        logger.info("Created models")
        logger.info(f"  Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def _create_ppo(self):
        """Crée l'algorithme PPO."""
        self.ppo = PPO(
            policy=self.policy,
            learning_rate=self.config.get("learning_rate", 3e-4),
            clip_epsilon=self.config.get("clip_epsilon", 0.2),
            value_clip_epsilon=self.config.get("value_clip_epsilon", 0.2),
            value_coef=self.config.get("value_coef", 0.5),
            entropy_coef=self.config.get("entropy_coef_start", 0.01),
            max_grad_norm=self.config.get("max_grad_norm", 0.5),
            ppo_epochs=self.config.get("ppo_epochs", 4),
            use_amp=self.config.get("amp_enabled", True),
            target_kl=self.config.get("target_kl", None),
            device=self.device,
        )
        
        logger.info("Created PPO algorithm")
    
    def _create_storage(self):
        """Crée le rollout storage."""
        self.storage = RolloutStorage(
            num_steps=self.config.get("rollout_len", 256),
            num_envs=self.config.get("num_envs", 48),
            obs_shape=(self.obs_dim,),
            action_space=self.envs.single_action_space,
            lstm_hidden_size=self.config.get("lstm_hidden_size", 128),
            lstm_num_layers=self.config.get("lstm_num_layers", 1),
            device=self.device,
        )
        
        logger.info("Created rollout storage")
    
    def _create_schedulers(self):
        """Crée les schedulers pour annealing."""
        # Entropy coefficient
        self.entropy_scheduler = AnnealingScheduler(
            start_value=self.config.get("entropy_coef_start", 0.01),
            end_value=self.config.get("entropy_coef_end", 0.002),
            total_steps=self.config.get("entropy_anneal_steps", 10000000),
            strategy="linear",
        )
        
        logger.info("Created schedulers")
    
    def _load_curriculum(self):
        """Charge le curriculum depuis la config."""
        # Charger le curriculum depuis la config (ou défaut)
        curriculum_file = self.config.get("curriculum_config", "configs/curriculum.yaml")
        curriculum_path = Path(curriculum_file)
        if curriculum_path.exists():
            with open(curriculum_path, "r") as f:
                curriculum_config = yaml.safe_load(f)
                self.curriculum_phases = curriculum_config.get("phases", [])
                logger.info(f"Loaded curriculum: {curriculum_file} ({len(self.curriculum_phases)} phases)")
        else:
            self.curriculum_phases = []
            logger.warning(f"Curriculum file not found: {curriculum_file}")
        
        logger.info(f"Loaded curriculum with {len(self.curriculum_phases)} phases")
    
    def _setup_logging(self):
        """Configure le logging (TensorBoard/W&B)."""
        from torch.utils.tensorboard import SummaryWriter
        
        # TensorBoard
        log_dir = Path("logs/tensorboard")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=120  # Flush automatique toutes les 120 secondes (réduit les I/O)
        )
        
        logger.info(f"TensorBoard logging enabled: {log_dir}")
        logger.info("Run: tensorboard --logdir logs/tensorboard")
    
    def _should_advance_curriculum(self) -> bool:
        """
        Vérifie si on doit passer à la phase suivante du curriculum.
        
        Returns:
            True si on doit avancer, False sinon
        """
        if not self.curriculum_phases or self.current_phase_idx >= len(self.curriculum_phases) - 1:
            return False  # Pas de curriculum ou déjà à la dernière phase
        
        current_phase = self.curriculum_phases[self.current_phase_idx]
        steps_in_phase = self.total_steps - self.phase_start_step
        
        # Initialiser stats (évite UnboundLocalError)
        stats = None
        
        # Vérifier les success criteria (PRIORITÉ)
        success_criteria = current_phase.get("success_criteria", {})
        if success_criteria and self.metrics_tracker is not None:
            stats = self.metrics_tracker.get_stats()
            if stats:
                # Vérifier qu'on a assez d'épisodes dans la fenêtre glissante
                num_episodes_in_window = len(self.metrics_tracker.wins) if hasattr(self.metrics_tracker, 'wins') else 0
                min_episodes = success_criteria.get("min_episodes", 50)
                
                # DEBUG: Log détaillé
                logger.debug(f"Curriculum check - Phase {self.current_phase_idx + 1}:")
                logger.debug(f"  Episodes in window: {num_episodes_in_window} (min required: {min_episodes})")
                logger.debug(f"  Current winrate: {stats.get('winrate', 0):.1f}% (target: {success_criteria.get('winrate_min', 0) * 100:.1f}%)")
                
                if num_episodes_in_window < min_episodes:
                    # Pas assez d'épisodes dans cette phase pour évaluer
                    logger.debug(f"  -> Not enough episodes yet, staying in phase")
                    return False
                
                # Vérifier winrate
                winrate_min = success_criteria.get("winrate_min", None)
                if winrate_min and stats.get('winrate', 0) >= winrate_min * 100:
                    logger.info(f"Curriculum: Phase {self.current_phase_idx + 1} success criteria met (winrate >= {winrate_min * 100}%)")
                    logger.info(f"  Evaluated on {num_episodes_in_window} episodes in this phase")
                    return True
                else:
                    logger.debug(f"  -> Winrate not reached yet, staying in phase")
        
        # Vérifier la durée (TIMEOUT seulement, pas critère d'avancement)
        duration_steps = current_phase.get("duration_steps", float('inf'))
        if steps_in_phase >= duration_steps:
            if success_criteria:
                # Il y a des critères de succès mais pas atteints
                logger.warning(f"Curriculum: Phase {self.current_phase_idx + 1} duration reached ({duration_steps} steps) but success criteria NOT met!")
                if stats:
                    logger.warning(f"  Current winrate: {stats.get('winrate', 0):.1f}% (target: {success_criteria.get('winrate_min', 0) * 100:.1f}%)")
                logger.warning(f"  Staying in current phase. Consider adjusting hyperparameters or rewards.")
                return False  # NE PAS AVANCER si critères pas atteints
            else:
                # Pas de critères de succès, avancer sur durée (WARNING: pas recommandé)
                logger.warning(f"Curriculum: Phase {self.current_phase_idx + 1} advancing without success criteria (duration: {duration_steps} steps)")
                logger.warning(f"  Consider adding success_criteria for safer curriculum progression")
                return True
        
        return False
    
    def _advance_curriculum(self):
        """Passe à la phase suivante du curriculum."""
        if self.current_phase_idx >= len(self.curriculum_phases) - 1:
            logger.info("Curriculum: Already at final phase")
            return
        
        self.current_phase_idx += 1
        self.phase_start_step = self.total_steps
        
        new_phase = self.curriculum_phases[self.current_phase_idx]
        new_env_name = new_phase.get("env_name", self.config.get("env_name"))
        
        logger.info("=" * 60)
        logger.info(f"CURRICULUM PHASE {self.current_phase_idx + 1}/{len(self.curriculum_phases)}: {new_phase.get('name', 'unnamed')}")
        logger.info(f"Environment: {new_env_name}")
        logger.info(f"Target: {new_phase.get('duration_steps', 'N/A')} steps")
        logger.info("=" * 60)
        
        # Mettre à jour la config avec le nouvel env
        self.config['env_name'] = new_env_name
        
        # Recréer les environnements avec le nouveau scénario
        logger.info("Recreating environments...")
        self.envs.close()
        self._create_env()
        
        # IMPORTANT : Recréer le storage avec les nouvelles dimensions d'observation
        logger.info("Recreating rollout storage...")
        self._create_storage()
        
        # RESET METRICS : Réinitialiser les statistiques pour la nouvelle phase
        if self.metrics_tracker is not None:
            logger.info("Resetting metrics for new phase...")
            self.metrics_tracker.reset()
        
        # RESET EPISODE TRACKERS : Réinitialiser les trackers individuels
        if self.episode_trackers is not None:
            logger.info("Resetting episode trackers...")
            env_name = self.config.get("env_name", "")
            self.episode_trackers = [DetailedEpisodeTracker(env_name=env_name) for _ in range(self.config.get("num_envs", 48))]
        
        # IMPORTANT : Recréer l'evaluator avec le nouvel environnement
        if self.evaluator is not None:
            logger.info("Recreating evaluator...")
            self.evaluator.close()
            
            eval_config = self.config.get("evaluation", {})
            self.evaluator = IntermediateEvaluator(
                eval_env=self.envs,
                policy=self.policy,
                device=self.device,
                eval_freq=eval_config.get("freq", 50000),
                num_eval_episodes=eval_config.get("num_episodes", 20),
                save_dir=eval_config.get("save_dir", "evaluations"),
            )
            logger.info("Evaluator recreated successfully")
        
        logger.info("Curriculum phase advanced successfully!")
    
    def _extract_positions_from_obs(self, env_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait les positions du joueur contrôlé et du ballon depuis l'environnement.
        
        ATTENTION: Cette méthode suppose que le wrapper expose get_original_obs().
        Si ce n'est pas le cas, elle retournera des positions par défaut [0.0, 0.0].
        
        Args:
            env_idx: Index de l'environnement à interroger
            
        Returns:
            player_pos [x, y], ball_pos [x, y]
        """
        try:
            # Vérifier si la méthode existe
            if not hasattr(self.envs, 'env_method'):
                logger.debug("env_method not available, returning default positions")
                return np.array([0.0, 0.0]), np.array([0.0, 0.0])
            
            # Appeler l'environnement pour obtenir les observations raw
            raw_obs = self.envs.env_method("get_original_obs", indices=[env_idx])[0]
            
            if raw_obs is None or not isinstance(raw_obs, dict):
                return np.array([0.0, 0.0]), np.array([0.0, 0.0])
            
            # Position du joueur contrôlé (premier joueur de left_team)
            left_team = raw_obs.get('left_team', None)
            if left_team is not None and len(left_team) > 0:
                player_pos = left_team[0][:2]  # [x, y]
            else:
                player_pos = np.array([0.0, 0.0])
            
            # Position du ballon
            ball = raw_obs.get('ball', None)
            if ball is not None and len(ball) >= 2:
                ball_pos = ball[:2]  # [x, y]
            else:
                ball_pos = np.array([0.0, 0.0])
            
            return player_pos, ball_pos
            
        except Exception as e:
            logger.debug(f"Failed to extract positions: {e}")
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    def train(self):
        """Boucle d'entraînement principale."""
        logger.info("Starting training...")
        
        # Afficher les infos du curriculum
        if self.curriculum_phases:
            current_phase = self.curriculum_phases[self.current_phase_idx]
            logger.info("=" * 60)
            logger.info(f"CURRICULUM PHASE {self.current_phase_idx + 1}/{len(self.curriculum_phases)}: {current_phase.get('name', 'unnamed')}")
            logger.info(f"Environment: {current_phase.get('env_name')}")
            logger.info(f"Target: {current_phase.get('duration_steps', 'N/A')} steps")
            if 'success_criteria' in current_phase:
                logger.info(f"Success criteria: {current_phase['success_criteria']}")
            logger.info("=" * 60)
        
        # Reset environnements (FIX: unpacking correct du tuple)
        obs, _ = self.envs.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        
        # Init LSTM state
        lstm_h = torch.zeros(
            self.config.get("lstm_num_layers", 1),
            self.config.get("num_envs", 48),
            self.config.get("lstm_hidden_size", 128),
            device=self.device,
        )
        lstm_c = torch.zeros_like(lstm_h)
        lstm_state = (lstm_h, lstm_c)
        
        # Stocker l'obs initiale
        self.storage.obs[0].copy_(obs)
        self.storage.lstm_h[0].copy_(lstm_h)
        self.storage.lstm_c[0].copy_(lstm_c)
        
        total_timesteps = self.config.get("total_timesteps", 50000000)
        rollout_len = self.config.get("rollout_len", 256)
        num_updates = total_timesteps // (rollout_len * self.config.get("num_envs", 48))
        
        start_time = time.time()
        
        for update in tqdm(range(num_updates), desc="Training"):
            # Collecter les rollouts
            for step in range(rollout_len):
                with torch.no_grad():
                    # Get action from policy
                    action, log_prob, value, lstm_state = self.policy.get_action(
                        obs, lstm_state, deterministic=False
                    )
                
                # Step dans l'environnement
                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                # Update football stats (besoin des obs brutes)
                # On va extraire les obs brutes depuis les envs
                try:
                    raw_obs_list = self.envs.call('get_original_obs')
                    for env_idx in range(self.config.get("num_envs", 48)):
                        if env_idx < len(raw_obs_list):
                            raw_obs = raw_obs_list[env_idx]
                            if raw_obs is not None:
                                self.football_stats.update_step(env_idx, raw_obs, action[env_idx].item(), info)
                except:
                    pass  # Si get_original_obs n'existe pas, skip
                
                # DEBUG : Log si reward > 0.5 (potentiel but)
                if reward.max() > 0.5:
                    env_idx_with_goal = reward.argmax()
                    # logger.info(f"🎯 BUT DETECTE! Env {env_idx_with_goal}, Reward={reward.max():.2f}")
                    # logger.info(f"   Info keys: {list(info.keys())}")
                    # Afficher score_reward si présent
                    # if 'score_reward' in info:
                    #     logger.info(f"   score_reward[{env_idx_with_goal}] = {info['score_reward'][env_idx_with_goal]}")
                    # if 'raw_score' in info:
                    #     logger.info(f"   raw_score[{env_idx_with_goal}] = {info['raw_score'][env_idx_with_goal]}")
                
                # Tracking des positions (heatmaps) - Optimisé : rotation entre envs
                if (self.position_tracker is not None or self.ball_tracker is not None) and step % 10 == 0:
                    # Rotation: tracker différents envs pour une meilleure couverture
                    env_idx = (step // 10) % self.config.get("num_envs", 48)
                    player_pos, ball_pos = self._extract_positions_from_obs(env_idx=env_idx)
                    
                    if self.position_tracker is not None:
                        self.position_tracker.update(player_pos)
                    
                    if self.ball_tracker is not None:
                        self.ball_tracker.update(ball_pos)
                
                # Tracking détaillé par environnement
                # SOLUTION : Extraire le score depuis les observations brutes stockées par l'env
                if self.episode_trackers is not None:
                    # Essayer d'extraire les scores depuis l'attribut de l'env si disponible
                    scores = None
                    if hasattr(self.envs, 'call'):
                        # AsyncVectorEnv a une méthode call pour appeler une méthode sur tous les envs
                        try:
                            scores = self.envs.call('get_current_score')
                        except:
                            pass
                    
                    for env_idx in range(self.config.get("num_envs", 48)):
                        # Extraire TOUTES les infos pour cet env
                        env_info = {}
                        
                        # Extraire toutes les clés de info pour cet env
                        for key, value in info.items():
                            if isinstance(value, (list, np.ndarray)) and len(value) > env_idx:
                                env_info[key] = value[env_idx]
                            elif not isinstance(value, (list, np.ndarray)):
                                # Valeur scalaire, copier telle quelle
                                env_info[key] = value
                        
                        # Ajouter score depuis call() si disponible
                        if scores is not None and env_idx < len(scores):
                            env_info["raw_score"] = scores[env_idx]
                        
                        self.episode_trackers[env_idx].step(
                            reward=reward[env_idx],
                            info=env_info
                        )
                
                next_obs = torch.from_numpy(next_obs).float().to(self.device)
                reward = torch.from_numpy(reward).float().to(self.device)
                done = torch.from_numpy(done).to(self.device)
                
                # Compute intrinsic rewards (RND curiosity)
                intrinsic_reward = None
                if self.use_rnd:
                    intrinsic_reward = self.rnd.compute_intrinsic_reward(next_obs)
                
                # Insérer dans storage
                self.storage.insert(
                    obs=next_obs,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                    lstm_state=lstm_state,
                    intrinsic_reward=intrinsic_reward,
                )
                
                # IMPORTANT : Reset LSTM states pour les environnements terminés
                if done.any():
                    lstm_h, lstm_c = lstm_state
                    done_mask = done.view(1, -1, 1).float()
                    lstm_h = lstm_h * (1.0 - done_mask)
                    lstm_c = lstm_c * (1.0 - done_mask)
                    lstm_state = (lstm_h, lstm_c)
                
                obs = next_obs
                self.total_steps += self.config.get("num_envs", 48)
                
                # Track rewards et lengths manuellement
                # reward vient de l'env (numpy) mais peut être converti en tensor
                if isinstance(reward, torch.Tensor):
                    reward_np = reward.cpu().numpy()
                else:
                    reward_np = np.array(reward, dtype=np.float32)
                self.episode_returns += reward_np
                self.episode_lengths += 1
                
                # Tracker les goals depuis les rewards bruts de GRF
                # Dans GRF: reward = +1 pour goal marqué, -1 pour goal concédé
                # Mais le RewardShaper modifie les rewards, donc on doit extraire le reward brut
                # On va tracker les goals à chaque step en regardant si done + reward
                # Approche: compter les goals au moment du done en regardant le score final
                
                # Tracking des épisodes terminés
                for env_idx in range(self.config.get("num_envs", 48)):
                    if done[env_idx]:
                        # Épisode terminé - sauvegarder les stats
                        episode_return = self.episode_returns[env_idx]
                        episode_length = self.episode_lengths[env_idx]
                        
                        self.episode_rewards.append(episode_return)
                        self.episode_lengths_deque.append(episode_length)
                        self.total_episodes_completed += 1
                        
                        # Football stats pour cet épisode
                        football_ep_stats = self.football_stats.on_episode_done(env_idx)
                        
                        # Extraire score depuis info (gère AsyncVectorEnv formats)
                        score_tuple = self._extract_score_from_info(info, env_idx, done)
                        
                        if score_tuple is not None:
                            goals_scored, goals_conceded = score_tuple
                            
                            # Update totals
                            self.total_goals_scored += goals_scored
                            self.total_goals_conceded += goals_conceded
                            
                            # Update W/D/L
                            if goals_scored > goals_conceded:
                                self.total_wins += 1
                            elif goals_scored < goals_conceded:
                                self.total_losses += 1
                            else:
                                self.total_draws += 1
                        else:
                            # Fallback: log warning première fois seulement
                            if self.total_episodes_completed == 1:
                                logger.warning("⚠️ raw_score not found in info - using fallback estimation")
                                logger.warning(f"   info keys: {list(info.keys()) if isinstance(info, dict) else 'not a dict'}")
                            
                            # Estimation depuis return (moins fiable)
                            self._estimate_results_from_return(episode_return)
                        
                        # Reset pour ce env
                        self.episode_returns[env_idx] = 0.0
                        self.episode_lengths[env_idx] = 0
            
            # Calculer la valeur du dernier état
            with torch.no_grad():
                next_value = self.policy.get_value(obs, lstm_state).squeeze(-1)
            
            # Calculer les returns et advantages
            use_intrinsic = self.use_rnd
            beta_intrinsic = self.config.get("rnd_beta", 0.1) if use_intrinsic else 0.0
            
            self.storage.compute_returns(
                next_value=next_value,
                gamma=self.config.get("gamma", 0.993),
                gae_lambda=self.config.get("lambda_gae", 0.95),
                use_intrinsic=use_intrinsic,
                beta_intrinsic=beta_intrinsic,
            )
            
            # Update PPO
            mini_batch_size = self.config.get("minibatch_size", None)
            metrics = self.ppo.update(
                storage=self.storage,
                mini_batch_size=mini_batch_size,
            )
            
            # Update RND (si activé)
            if self.use_rnd:
                # Flatten all observations from storage
                all_obs = self.storage.obs[:-1].reshape(-1, self.obs_dim)
                rnd_loss = self.rnd.update(all_obs)
                metrics["rnd_loss"] = rnd_loss
                metrics["intrinsic_reward_mean"] = self.storage.intrinsic_rewards.mean().item()
            
            # Après update
            self.storage.after_update()
            
            # Update reward shaper step
            self.reward_shaper.update_step(self.total_steps)
            
            # Update entropy coef
            new_entropy_coef = self.entropy_scheduler.step()
            self.ppo.set_entropy_coef(new_entropy_coef)
            
            # Logging
            if update % self.config.get("log_interval", 10) == 0:
                self._log_metrics(update, metrics)
            
            # Évaluation intermédiaire
            if self.evaluator is not None and self.evaluator.should_evaluate(self.total_steps):
                eval_results = self.evaluator.evaluate(self.total_steps)
                
                # Logger dans TensorBoard
                if hasattr(self, 'writer'):
                    for key, value in eval_results.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"eval/{key}", value, self.total_steps)
                
                # Afficher le résumé
                logger.info("\n" + self.evaluator.get_summary_str())
            
            # Afficher les métriques de goals et winrate
            if update % self.config.get("log_interval", 10) == 0 and self.total_episodes_completed > 0:
                winrate = (self.total_wins / self.total_episodes_completed) * 100
                drawrate = (self.total_draws / self.total_episodes_completed) * 100
                lossrate = (self.total_losses / self.total_episodes_completed) * 100
                goals_per_game = self.total_goals_scored / self.total_episodes_completed
                conceded_per_game = self.total_goals_conceded / self.total_episodes_completed
                
                logger.info("\n📊 TRAINING METRICS (last 100 episodes)")
                logger.info(f"\n⚽ Goals:")
                logger.info(f"  Scored:    {goals_per_game:.2f}/game")
                logger.info(f"  Conceded:  {conceded_per_game:.2f}/game")
                logger.info(f"\n🏆 Results:")
                logger.info(f"  Winrate:   {winrate:.1f}%")
                logger.info(f"  Drawrate:  {drawrate:.1f}%")
                logger.info(f"  Lossrate:  {lossrate:.1f}%")
                logger.info(f"\n🎯 Total:")
                logger.info(f"  Episodes:  {self.total_episodes_completed}")
                logger.info(f"  Wins:      {self.total_wins}")
                logger.info(f"  Goals:     {self.total_goals_scored}\n")
            
            # Vérifier le curriculum (changement de phase automatique)
            if self._should_advance_curriculum():
                self._advance_curriculum()
                
                # FIX: Reset LSTM states ET observations après changement d'environnement
                # _advance_curriculum() a déjà fermé et recréé les envs, donc on reset
                obs, _ = self.envs.reset()
                obs = torch.from_numpy(obs).float().to(self.device)
                
                # Reset LSTM
                lstm_h.zero_()
                lstm_c.zero_()
                lstm_state = (lstm_h, lstm_c)
                
                # Initialiser le storage avec la nouvelle observation
                self.storage.obs[0].copy_(obs)
                self.storage.lstm_h[0].copy_(lstm_h)
                self.storage.lstm_c[0].copy_(lstm_c)
            
            # Checkpointing
            if update % (self.config.get("save_freq", 100000) // (rollout_len * self.config.get("num_envs", 48))) == 0:
                self._save_checkpoint("last.pt")
                
                # Save best
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards)
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self._save_checkpoint("best.pt")
                
                # Sauvegarder l'historique des métriques
                if self.metrics_tracker is not None:
                    self.metrics_tracker.save_history("logs/metrics_history.json")
        
        logger.info("Training completed!")
        elapsed_time = time.time() - start_time
        logger.info(f"Total time: {elapsed_time:.2f}s")
    
    def _log_metrics(self, update: int, ppo_metrics: Dict[str, float]):
        """Log les métriques."""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths_deque)
            
            logger.info(f"Update {update} | Steps {self.total_steps}")
            logger.info(f"  Mean reward: {mean_reward:.2f}")
            logger.info(f"  Mean length: {mean_length:.1f}")
            logger.info(f"  Policy loss: {ppo_metrics['policy_loss']:.4f}")
            logger.info(f"  Value loss: {ppo_metrics['value_loss']:.4f}")
            logger.info(f"  Entropy: {ppo_metrics['entropy']:.4f}")
            if self.use_rnd:
                logger.info(f"  RND loss: {ppo_metrics.get('rnd_loss', 0):.4f}")
                logger.info(f"  Intrinsic reward: {ppo_metrics.get('intrinsic_reward_mean', 0):.4f}")
            
            # Logger dans TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_scalar("train/mean_reward", mean_reward, self.total_steps)
                self.writer.add_scalar("train/mean_length", mean_length, self.total_steps)
                self.writer.add_scalar("train/policy_loss", ppo_metrics['policy_loss'], self.total_steps)
                self.writer.add_scalar("train/value_loss", ppo_metrics['value_loss'], self.total_steps)
                self.writer.add_scalar("train/entropy", ppo_metrics['entropy'], self.total_steps)
                
                # Stats de goals et winrate
                if self.total_episodes_completed > 0:
                    winrate = (self.total_wins / self.total_episodes_completed) * 100
                    goals_per_game = self.total_goals_scored / self.total_episodes_completed
                    conceded_per_game = self.total_goals_conceded / self.total_episodes_completed
                    
                    self.writer.add_scalar("results/winrate", winrate, self.total_steps)
                    self.writer.add_scalar("goals/scored_per_game", goals_per_game, self.total_steps)
                    self.writer.add_scalar("goals/conceded_per_game", conceded_per_game, self.total_steps)
                    self.writer.add_scalar("goals/total_scored", self.total_goals_scored, self.total_steps)
                    self.writer.add_scalar("goals/total_conceded", self.total_goals_conceded, self.total_steps)
                    self.writer.add_scalar("goals/own_goals", self.total_own_goals, self.total_steps)
                
                # Football stats (possession, tirs, passes, etc.)
                football_stats = self.football_stats.get_summary_stats()
                if football_stats:
                    self.writer.add_scalar("football/possession_pct", football_stats.get('possession_pct', 0), self.total_steps)
                    self.writer.add_scalar("football/shots_per_game", football_stats.get('shots_per_game', 0), self.total_steps)
                    self.writer.add_scalar("football/shots_on_target_per_game", football_stats.get('shots_on_target_per_game', 0), self.total_steps)
                    self.writer.add_scalar("football/shot_accuracy", football_stats.get('shot_accuracy', 0), self.total_steps)
                    self.writer.add_scalar("football/passes_per_game", football_stats.get('passes_per_game', 0), self.total_steps)
                    self.writer.add_scalar("football/pass_accuracy", football_stats.get('pass_accuracy', 0), self.total_steps)
                    self.writer.add_scalar("football/distance_traveled", football_stats.get('distance_traveled', 0), self.total_steps)
                    self.writer.add_scalar("football/time_in_defense", football_stats.get('time_in_defense', 0), self.total_steps)
                    self.writer.add_scalar("football/time_in_midfield", football_stats.get('time_in_midfield', 0), self.total_steps)
                    self.writer.add_scalar("football/time_in_attack", football_stats.get('time_in_attack', 0), self.total_steps)
                
                # Heatmaps (toutes les 5000 steps)
                if self.total_steps % 5000 == 0:
                    heatmaps = self.football_stats.get_heatmaps()
                    if heatmaps:
                        # Convertir en images pour TensorBoard
                        import matplotlib.pyplot as plt
                        import io
                        from PIL import Image
                        
                        # Player heatmap
                        fig, ax = plt.subplots(figsize=(10, 7))
                        im = ax.imshow(heatmaps['player'].T, cmap='hot', origin='lower', aspect='auto')
                        ax.set_title('Player Heatmap')
                        ax.set_xlabel('X (field)')
                        ax.set_ylabel('Y (field)')
                        plt.colorbar(im, ax=ax)
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        img = Image.open(buf)
                        img_array = np.array(img)
                        self.writer.add_image('heatmap/player', img_array, self.total_steps, dataformats='HWC')
                        plt.close(fig)
                        buf.close()
                        
                        # Ball heatmap
                        fig, ax = plt.subplots(figsize=(10, 7))
                        im = ax.imshow(heatmaps['ball'].T, cmap='hot', origin='lower', aspect='auto')
                        ax.set_title('Ball Heatmap')
                        ax.set_xlabel('X (field)')
                        ax.set_ylabel('Y (field)')
                        plt.colorbar(im, ax=ax)
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        img = Image.open(buf)
                        img_array = np.array(img)
                        self.writer.add_image('heatmap/ball', img_array, self.total_steps, dataformats='HWC')
                        plt.close(fig)
                        buf.close()
                        
                        # Reset heatmaps pour éviter saturation
                        self.football_stats.reset_heatmaps()
                
                # Métriques avancées (désactivé)
                if False and self.metrics_tracker is not None:
                    stats = self.metrics_tracker.get_stats()
                    if stats:
                        # Résultats (%)
                        self.writer.add_scalar("results/winrate", stats.get('winrate', 0), self.total_steps)
                        self.writer.add_scalar("results/drawrate", stats.get('drawrate', 0), self.total_steps)
                        self.writer.add_scalar("results/lossrate", stats.get('lossrate', 0), self.total_steps)
                        
                        # Buts (moyennes glissantes)
                        self.writer.add_scalar("goals/scored_avg", stats.get('avg_goals_scored', 0), self.total_steps)
                        self.writer.add_scalar("goals/conceded_avg", stats.get('avg_goals_conceded', 0), self.total_steps)
                        self.writer.add_scalar("goals/difference", stats.get('goal_difference', 0), self.total_steps)
                        self.writer.add_scalar("goals/own_goals_avg", stats.get('avg_own_goals', 0), self.total_steps)
                        
                        # Buts (totaux cumulatifs)
                        self.writer.add_scalar("cumulative/total_goals_scored", stats.get('total_goals_scored', 0), self.total_steps)
                        self.writer.add_scalar("cumulative/total_goals_conceded", stats.get('total_goals_conceded', 0), self.total_steps)
                        self.writer.add_scalar("cumulative/total_own_goals", stats.get('total_own_goals', 0), self.total_steps)
                        self.writer.add_scalar("cumulative/total_episodes", stats.get('total_episodes', 0), self.total_steps)
                        
                        # Performance
                        self.writer.add_scalar("performance/avg_return", stats.get('avg_return', 0), self.total_steps)
                        self.writer.add_scalar("performance/avg_episode_length", stats.get('avg_episode_length', 0), self.total_steps)
                        
                        # Ratio de buts par match (indicateur d'efficacité)
                        if stats.get('avg_episode_length', 0) > 0:
                            goals_per_100steps = stats.get('avg_goals_scored', 0) / (stats.get('avg_episode_length', 1) / 100)
                            self.writer.add_scalar("efficiency/goals_per_100steps", goals_per_100steps, self.total_steps)
                
                # Heatmaps et métriques de position
                if self.position_tracker is not None and self.total_steps % self.heatmap_freq == 0:
                    try:
                        # Générer et logger la heatmap des positions
                        position_heatmap = self.position_tracker.get_heatmap_image()
                        self.writer.add_image("heatmaps/player_position", position_heatmap, 
                                             self.total_steps, dataformats='HWC')
                        
                        # Métriques de position
                        pos_metrics = self.position_tracker.get_metrics()
                        for key, value in pos_metrics.items():
                            self.writer.add_scalar(key, value, self.total_steps)
                        
                        logger.info(f"  📊 Heatmap generated at step {self.total_steps}")
                        logger.info(f"  Position coverage: {pos_metrics.get('position/coverage', 0):.1f}%")
                        logger.info(f"  Offensive ratio: {pos_metrics.get('position/offensive_ratio', 0):.1f}%")
                    except Exception as e:
                        logger.warning(f"Failed to generate player heatmap: {e}")
                
                # Heatmap du ballon
                if self.ball_tracker is not None and self.total_steps % self.heatmap_freq == 0:
                    try:
                        ball_heatmap = self.ball_tracker.get_heatmap_image()
                        self.writer.add_image("heatmaps/ball_position", ball_heatmap,
                                             self.total_steps, dataformats='HWC')
                        
                        # Métriques du ballon
                        ball_metrics = self.ball_tracker.get_metrics()
                        for key, value in ball_metrics.items():
                            self.writer.add_scalar(key, value, self.total_steps)
                        
                        logger.info(f"  ⚽ Ball heatmap generated at step {self.total_steps}")
                    except Exception as e:
                        logger.warning(f"Failed to generate ball heatmap: {e}")
    
    def _extract_score_from_info(self, info: Dict, env_idx: int, done_mask: np.ndarray) -> Optional[tuple]:
        """
        Extrait le score [goals_scored, goals_conceded] depuis info.
        Gère les 3 formats possibles de gymnasium.vector.AsyncVectorEnv.
        
        Returns:
            (goals_scored, goals_conceded) ou None si pas trouvé
        """
        # Format 1: Gymnasium AsyncVectorEnv avec final_info (quand done=True)
        if done_mask[env_idx] and isinstance(info, dict) and 'final_info' in info:
            final_infos = info['final_info']
            if isinstance(final_infos, (list, np.ndarray)) and len(final_infos) > env_idx:
                final_info = final_infos[env_idx]
                if final_info is not None and isinstance(final_info, dict) and 'raw_score' in final_info:
                    score = final_info['raw_score']
                    if isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                        return (int(score[0]), int(score[1]))
        
        # Format 2: info[env_idx]['raw_score'] (custom format)
        if isinstance(info, dict) and env_idx in info:
            env_info = info[env_idx]
            if isinstance(env_info, dict) and 'raw_score' in env_info:
                score = env_info['raw_score']
                if isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                    return (int(score[0]), int(score[1]))
        
        # Format 3: info['raw_score'][env_idx] (vectorized format)
        if isinstance(info, dict) and 'raw_score' in info:
            raw_scores = info['raw_score']
            if hasattr(raw_scores, '__getitem__') and len(raw_scores) > env_idx:
                score = raw_scores[env_idx]
                if isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                    return (int(score[0]), int(score[1]))
        
        return None
    
    def _estimate_results_from_return(self, episode_return: float):
        """Fallback: estime les résultats depuis le return (moins fiable à cause du shaping)."""
        # Cette méthode est un fallback si on n'a pas accès au score GRF
        # Elle est moins fiable car le return inclut le shaping
        if episode_return >= 15:  # Probablement un goal
            self.total_goals_scored += 1
            self.total_wins += 1
        elif episode_return <= -15:  # Probablement CSC ou goal concédé
            if episode_return <= -18:
                self.total_own_goals += 1
            else:
                self.total_goals_conceded += 1
            self.total_losses += 1
        else:  # Draw
            self.total_draws += 1
    
    def _save_checkpoint(self, filename: str):
        """Sauvegarde un checkpoint avec infos curriculum et hyperparams."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        # Sauvegarder les poids via PPO
        self.ppo.save_checkpoint(str(checkpoint_path))
        
        # Charger pour ajouter les infos curriculum et hyperparams
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["total_steps"] = self.total_steps
        checkpoint["current_phase_idx"] = self.current_phase_idx
        checkpoint["phase_start_step"] = self.phase_start_step
        checkpoint["best_mean_reward"] = self.best_mean_reward
        
        # FIX: Sauvegarder les hyperparams du modèle pour compatibilité
        checkpoint["model_config"] = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.config.get("hidden_size", 512),
            "lstm_hidden_size": self.config.get("lstm_hidden_size", 128),
            "lstm_num_layers": self.config.get("lstm_num_layers", 1),
        }
        
        # Sauvegarder RND (si activé)
        if self.use_rnd:
            checkpoint["rnd"] = self.rnd.state_dict()
            checkpoint["use_rnd"] = True
        else:
            checkpoint["use_rnd"] = False
        
        # Re-sauvegarder avec infos complètes
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path} (phase {self.current_phase_idx + 1}/{len(self.curriculum_phases) if self.curriculum_phases else 'N/A'})")
    
    def _load_checkpoint(self, filename: str):
        """Charge un checkpoint avec restauration du curriculum."""
        checkpoint_path = Path("checkpoints") / filename
        
        # Charger via PPO
        self.ppo.load_checkpoint(str(checkpoint_path))
        
        # Charger les infos curriculum
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
            logger.info(f"Restored total_steps: {self.total_steps}")
        
        if "current_phase_idx" in checkpoint:
            old_phase = self.current_phase_idx
            self.current_phase_idx = checkpoint["current_phase_idx"]
            self.phase_start_step = checkpoint.get("phase_start_step", 0)
            
            # Mettre à jour l'environnement si nécessaire
            if self.curriculum_phases and self.current_phase_idx < len(self.curriculum_phases):
                phase = self.curriculum_phases[self.current_phase_idx]
                phase_env = phase.get("env_name")
                
                if phase_env and phase_env != self.config.get("env_name"):
                    logger.info(f"Restoring curriculum phase {self.current_phase_idx + 1}/{len(self.curriculum_phases)}: {phase.get('name')}")
                    self.config['env_name'] = phase_env
                    
                    # Recréer les envs avec le bon scénario
                    logger.info(f"Recreating environments for {phase_env}...")
                    self.envs.close()
                    self._create_env()
                    
                logger.info(f"Curriculum phase restored: {self.current_phase_idx + 1}/{len(self.curriculum_phases)}")
            else:
                logger.info(f"Restored curriculum phase: {self.current_phase_idx + 1}")
        
        if "best_mean_reward" in checkpoint:
            self.best_mean_reward = checkpoint["best_mean_reward"]
        
        # Charger RND (si présent et activé)
        if self.use_rnd and checkpoint.get("use_rnd", False):
            if "rnd" in checkpoint:
                self.rnd.load_state_dict(checkpoint["rnd"])
                logger.info("Loaded RND state from checkpoint")
            else:
                logger.warning("RND is enabled but no RND state found in checkpoint")
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")