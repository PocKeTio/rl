"""
Script d'entraînement principal pour PPO+LSTM+RND.

Usage:
    python -m gfrl.cli.train_ppo_rnd
    python -m gfrl.cli.train_ppo_rnd --config configs/train_ppo_rnd.yaml
"""

import argparse
from pathlib import Path
import yaml

from ..train.trainer import Trainer
from ..utils.logging import setup_logger

logger = setup_logger("train")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO+LSTM+RND agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_ppo_rnd.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Charger la config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 60)
    logger.info("Starting PPO+LSTM+RND Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Total timesteps: {config.get('total_timesteps', 50000000):,}")
    logger.info(f"Num envs: {config.get('num_envs', 48)}")  # Default cohérent avec trainer.py
    logger.info(f"Device: {config.get('device', 'cuda')}")
    logger.info("=" * 60)
    
    # Créer le trainer
    trainer = Trainer(config_dict=config)
    
    # Resume si spécifié
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer._load_checkpoint(Path(args.resume).name)
    
    # Lancer l'entraînement
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint("interrupted.pt")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        trainer._save_checkpoint("error.pt")
        raise
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
