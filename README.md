# Google Research Football - PPO + RND

Implémentation PPO (Proximal Policy Optimization) avec RND (Random Network Distillation) pour l'environnement Google Research Football.

## 🎯 Features

- **PPO Algorithm** avec LSTM pour mémoire temporelle
- **RND (Curiosity-driven exploration)** pour sparse rewards
- **Reward Shaping** configurable (dense/sparse)
- **Curriculum Learning** de 3v1 à 5v5
- **AMP (Automatic Mixed Precision)** pour optimisation GPU
- **Vectorized Environments** (AsyncVectorEnv)
- **TensorBoard logging** avec métriques détaillées

## 🏗️ Architecture

```
src/gfrl/
├── algo/           # PPO + Storage (GAE)
├── models/         # Policy LSTM + RND
├── train/          # Trainer avec curriculum
├── env/            # Wrappers GRF + Reward Shaper
├── utils/          # Logging, encoders, schedulers
└── cli/            # Entry point training
```

## 📦 Installation

```bash
# Créer environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

## 🚀 Training

### Sparse Rewards (avec RND)
```powershell
.\START_SPARSE.ps1
```

### Dense Rewards (reward shaping)
```powershell
python -m src.gfrl.cli.train_ppo_rnd --config configs/train_5v5.yaml
```

### Continuer training
```powershell
.\CONTINUE_SPARSE.ps1
```

## ⚙️ Configuration

### Train Config (`configs/train_sparse.yaml`)
```yaml
# PPO Hyperparameters
learning_rate: 0.0003
clip_epsilon: 0.2
gamma: 0.997
lambda_gae: 0.95

# RND (Curiosity)
use_rnd: true
rnd_beta: 0.05  # Intrinsic reward weight

# Architecture
hidden_size: 512
lstm_hidden_size: 128
lstm_num_layers: 1
```

### Rewards (`configs/rewards_sparse.yaml`)
```yaml
# Sparse: Goals only + minimal guidance
goal_scored: 100.0
goal_distance:
  enabled: true
  reward_scale: 0.5  # Minimal guidance
```

### Curriculum (`configs/curriculum_5v5.yaml`)
7 phases progressives:
- Phase 1-3: academy (3v1, 3v2, 3v3)
- Phase 4: 1_vs_1_easy
- Phase 5-7: 5v5 (easy → medium → hard)

## 📊 Monitoring

TensorBoard:
```bash
tensorboard --logdir=logs
```

Visualiser agent:
```bash
python scripts/watch_agent.py --checkpoint checkpoints/last.pt --render
```

## 🔧 Fixes Appliqués

### RND Normalization
- **Bug:** Formule std incorrecte
- **Fix:** EMA du std réel du batch MSE

### LSTM Memory
- **Bug:** Mini-batch reset states
- **Fix:** Full batch pour préserver mémoire

### AsyncVectorEnv Info
- **Bug:** Format info non géré
- **Fix:** Extraction robuste avec 3 formats

### Hyperparameters
- `rnd_beta`: 0.5 → 0.05 (task dominant)
- `entropy_coef`: 0.1 → 0.02 (équilibré)
- `clip_epsilon`: 0.12 → 0.2 (standard PPO)
- `learning_rate`: 0.0001 → 0.0003 (standard)

## 📈 Résultats attendus

```
0-5M steps:    Exploration + premiers goals
5-15M steps:   Patterns émergent (10-20% winrate)
15-30M steps:  Stratégie basique (25-40% winrate)
30-50M steps:  Maîtrise 3v1 (50-70% winrate)
50M+ steps:    Transfer vers 5v5
```

## 📚 Références

- **PPO:** Schulman et al. (2017) - Proximal Policy Optimization
- **RND:** Burda et al. (2018) - Exploration by Random Network Distillation
- **GRF:** Kurach et al. (2020) - Google Research Football
- **GAE:** Schulman et al. (2015) - Generalized Advantage Estimation

## 📝 Documentation

Voir `docs/`:
- `ARCHITECTURE_THEORY.md`: Architecture détaillée
- `RND_IMPLEMENTATION.md`: Implémentation RND
- `SPARSE_REWARDS_PROBLEM.md`: Problèmes sparse RL
- `INFO_FORMAT_FIXES.md`: Fixes AsyncVectorEnv

## 🎮 Environnements supportés

- `academy_3_vs_1_with_keeper`
- `academy_3_vs_2_with_keeper`
- `academy_3_vs_3_with_keeper`
- `1_vs_1_easy`
- `5_vs_5_easy`
- `5_vs_5_medium`
- `5_vs_5_hard`

## 🔬 Expérimentations

### Sparse vs Dense
- **Sparse + RND:** Exploration autonome, apprentissage lent mais robuste
- **Dense:** Convergence rapide, risque de sur-optimisation du shaping

### RND Beta
- `0.01`: Très conservateur (task >> curiosité)
- `0.05`: Équilibré (recommandé)
- `0.1+`: Curiosité domine (agent explore sans apprendre task)

## ⚠️ Problèmes connus

1. **Entropy élevée pendant bootstrap:** Normal avec sparse rewards
2. **Compteurs goals:** Nécessite `raw_score` dans info (géré)
3. **LSTM full batch:** Nécessaire pour préserver mémoire temporelle

## 🚀 Roadmap

- [ ] Self-play contre anciennes policies
- [ ] Population-based training (PBT)
- [ ] Attention mechanism dans policy
- [ ] Opponent modeling
- [ ] Multi-agent coordination

## 📄 License

MIT

## 🙏 Acknowledgments

- Google Research Football team
- OpenAI PPO implementation
- Burda et al. pour RND
