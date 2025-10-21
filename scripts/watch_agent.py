"""
Script pour observer un agent jouer contre l'IA intégrée.

Usage:
    python scripts/watch_agent.py --checkpoint checkpoints/last.pt
    python scripts/watch_agent.py --checkpoint checkpoints/last.pt --num-games 5 --render
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time

from gfrl.models.policy_lstm import PolicyLSTM
from gfrl.utils.obs_encoder import RawObsEncoder
from gfrl.env.make_env import create_grf_env
from gfrl.env.rewarders import RewardShaper
import gfootball.env as football_env


def watch_agent(checkpoint_path: str, num_games: int = 3, render: bool = True, env_name: str = "5_vs_5_medium", stochastic: bool = False, use_training_wrapper: bool = True, reward_config: str = "configs/rewards_dense.yaml"):
    """
    Fait jouer l'agent et affiche les stats.
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        num_games: Nombre de matchs à jouer
        render: Si True, affiche le rendu graphique
        env_name: Nom de l'environnement GRF
    """
    print("=" * 70)
    print("WATCH AGENT - Observation de l'agent")
    print("=" * 70)
    
    # Charger le checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n[OK] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Créer la policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyLSTM(obs_dim=161, action_dim=19)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device)
    policy.eval()
    
    print(f"[OK] Policy loaded on {device}")
    
    # Info du checkpoint
    if "total_steps" in checkpoint:
        total_steps = checkpoint["total_steps"]
        print(f"   - Total training steps: {total_steps:,} ({total_steps / 1e6:.1f}M)")
    
    if "current_phase_idx" in checkpoint:
        phase_idx = checkpoint["current_phase_idx"]
        print(f"   - Curriculum phase: {phase_idx + 1}")
    
    # Créer l'environnement
    print(f"\n[ENV] Creating environment: {env_name}")
    if use_training_wrapper:
        shaper = RewardShaper(config_path=Path(reward_config))
        env = create_grf_env(
            env_name=env_name,
            representation='raw',
            action_set='sticky',
            frame_skip=1,  # MUST MATCH train_5v5.yaml!
            render=render,
            obs_wrapper=True,
            running_norm=False,  # MUST MATCH train_5v5.yaml!
            mirror=False,
            reward_shaper=shaper,
        )
        print(f"[OK] Environment created (training wrappers)")
        print(f"   - Render: {'ON' if render else 'OFF'}")
        print(f"   - Action space: {env.action_space.n} actions")
        print(f"[OK] Observation encoder created (dim={env.observation_space.shape[0]})")
    else:
        # Direct GRF (fallback)
        env = football_env.create_environment(
            env_name=env_name,
            representation='raw',
            stacked=False,
            logdir='',
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=render,
            write_video=False,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
        )
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped
        print(f"[OK] Environment created (raw GRF)")
        print(f"   - Render: {'ON' if render else 'OFF'}")
        print(f"   - Action space: {env.action_space.n} actions")
        obs_encoder = RawObsEncoder()
        print(f"[OK] Observation encoder created (dim={obs_encoder.obs_dim})")
    
    # Statistiques
    wins = 0
    draws = 0
    losses = 0
    total_goals_scored = 0
    total_goals_conceded = 0
    total_steps = 0
    
    print(f"\n{'=' * 70}")
    print(f"Playing {num_games} game(s)...")
    print(f"{'=' * 70}\n")
    
    for game_idx in range(num_games):
        print(f"Game {game_idx + 1}/{num_games}")
        
        if use_training_wrapper:
            obs, _ = env.reset()
        else:
            obs = env.reset()
        done = False
        
        # LSTM state
        lstm_h = torch.zeros(1, 1, 128, device=device)
        lstm_c = torch.zeros(1, 1, 128, device=device)
        lstm_state = (lstm_h, lstm_c)
        
        game_steps = 0
        goals_scored = 0
        goals_conceded = 0
        
        while not done:
            # Observation encodée
            if use_training_wrapper:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                if isinstance(obs, list):
                    obs_dict = obs[0]
                else:
                    obs_dict = obs
                obs_encoded = obs_encoder.encode(obs_dict)
                obs_tensor = torch.FloatTensor(obs_encoded).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value, lstm_state = policy.get_action(
                    obs_tensor, lstm_state, deterministic=not stochastic
                )
            
            # Step
            if use_training_wrapper:
                step_result = env.step(action.item())
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated) or bool(truncated)
                else:
                    obs, reward, done, info = step_result
            else:
                obs, reward, done, info = env.step(action.item())
            
            # Track score (prefer base reward/score indicators)
            if isinstance(reward, list):
                reward = reward[0]
            base_reward = None
            if isinstance(info, dict):
                for key in ("reward_base", "_reward_base", "score_reward", "_score_reward"):
                    if key in info:
                        try:
                            base_reward = float(info[key])
                            break
                        except (TypeError, ValueError):
                            pass
            if base_reward is not None:
                if base_reward >= 0.9:
                    goals_scored += 1
                    print(f"   Goal! {goals_scored}-{goals_conceded}")
                elif base_reward <= -0.9:
                    goals_conceded += 1
                    print(f"   Conceded... {goals_scored}-{goals_conceded}")
            else:
                if reward > 0.5:
                    goals_scored += 1
                    print(f"   Goal! {goals_scored}-{goals_conceded}")
                elif reward < -0.5:
                    goals_conceded += 1
                    print(f"   Conceded... {goals_scored}-{goals_conceded}")
            
            game_steps += 1
            
            if render:
                time.sleep(0.01)  # Slow down for visibility
        
        # Game ended
        total_steps += game_steps
        total_goals_scored += goals_scored
        total_goals_conceded += goals_conceded
        
        if goals_scored > goals_conceded:
            wins += 1
            result = "[WIN]"
        elif goals_scored < goals_conceded:
            losses += 1
            result = "[LOSS]"
        else:
            draws += 1
            result = "[DRAW]"
        
        print(f"   {result} - Final score: {goals_scored}-{goals_conceded} ({game_steps} steps)")
        print()
    
    # Final stats
    print(f"{'=' * 70}")
    print(f"FINAL STATISTICS ({num_games} games)")
    print(f"{'=' * 70}")
    print(f"\nResults:")
    print(f"   - Wins:   {wins} ({wins/num_games*100:.1f}%)")
    print(f"   - Draws:  {draws} ({draws/num_games*100:.1f}%)")
    print(f"   - Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"\nGoals:")
    print(f"   - Scored:   {total_goals_scored} ({total_goals_scored/num_games:.2f}/game)")
    print(f"   - Conceded: {total_goals_conceded} ({total_goals_conceded/num_games:.2f}/game)")
    print(f"   - Diff:     {total_goals_scored - total_goals_conceded:+d}")
    print(f"\nAverage game length: {total_steps/num_games:.0f} steps")
    
    # Diagnostic
    print(f"\nDiagnostic:")
    winrate = wins / num_games * 100
    
    if winrate < 10:
        print(f"   [CRITICAL] Winrate < 10% - Agent ne fonctionne pas")
    elif winrate < 30:
        print(f"   [WARNING] Winrate < 30% - Apprentissage insuffisant")
    elif winrate < 50:
        print(f"   [FAIR] Winrate {winrate:.0f}% - Progression necessaire")
    elif winrate < 70:
        print(f"   [GOOD] Winrate {winrate:.0f}% - Bon niveau")
    else:
        print(f"   [EXCELLENT] Winrate {winrate:.0f}% - Tres bon niveau")
    
    if total_goals_scored / num_games < 0.1:
        print(f"   [WARNING] Tres peu de buts marques ({total_goals_scored/num_games:.2f}/game)")
        print(f"      -> L'agent ne tire pas ou ne sait pas marquer")
    
    if draws / num_games > 0.8:
        print(f"   [WARNING] Trop de matchs nuls ({draws/num_games*100:.0f}%)")
        print(f"      -> L'agent timeout sans marquer")
    
    print(f"\n{'=' * 70}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Watch an agent play")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    parser.add_argument("--num-games", type=int, default=3,
                       help="Number of games to play")
    parser.add_argument("--render", action="store_true",
                       help="Render the game visually")
    parser.add_argument("--env", type=str, default="5_vs_5_medium",
                       help="Environment name")
    parser.add_argument("--stochastic", action="store_true",
                       help="Sample actions stochastically (train-like)")
    parser.add_argument("--use-training-wrapper", action="store_true",
                       help="Use same wrappers as training (RewardShaper, ObsWrapperRaw)")
    parser.add_argument("--reward-config", type=str, default="configs/rewards_dense.yaml",
                       help="Reward config path (used with --use-training-wrapper)")
    args = parser.parse_args()
    
    watch_agent(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        render=args.render,
        env_name=args.env,
        stochastic=args.stochastic,
        use_training_wrapper=args.use_training_wrapper,
        reward_config=args.reward_config,
    )


if __name__ == "__main__":
    main()
