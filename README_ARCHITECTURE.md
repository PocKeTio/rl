# 🏆 Google Research Football - Architecture du Projet

Ce document détaille les fichiers utilisés dans le pipeline d'entraînement et leur rôle.

---

## 📋 Table des matières

- [Configs](#configs)
- [Code source](#code-source)
- [Scripts](#scripts)
- [Pipeline d'exécution](#pipeline-dexécution)

---

## 📁 Configs

Fichiers YAML de configuration situés dans `configs/`.

### `train_5v5.yaml`
**Rôle**: Configuration principale de l'entraînement PPO

**Contenu**:
- Hyperparamètres PPO (learning rate, clip_epsilon, gamma, GAE lambda)
- Entropy annealing (0.01 → 0.001 sur 5M steps)
- Batch size et architecture (num_envs, rollout_len, ppo_epochs)
- RND (exploration intrinsèque, actuellement désactivé)
- Logging et checkpointing

**Utilisé par**: `src/gfrl/train/trainer.py`

---

### `env.yaml`
**Rôle**: Configuration de l'environnement GRF

**Contenu**:
- Nom du scénario (`academy_empty_goal_close`)
- Type de représentation (`raw` - observations brutes)
- Action set (`sticky` - 19 actions)
- Nombre d'environnements parallèles (`num_envs: 24`)
- Frame skip et paramètres de rendering

**Utilisé par**: `src/gfrl/env/make_env.py`

---

### `rewards_dense.yaml`
**Rôle**: Configuration du reward shaping dense

**Contenu**:
- **Sparse rewards**: goal_scored (+5.0), own_goal (-10.0), win_bonus (+2.0)
- **Dense rewards**: 
  - `goal_distance` (+0.08/step) - Guide vers le but
  - `penalty_box_bonus` (+0.15/step) - Encourage entrée surface
  - `shot_attempt` (+1.5 si x≥0.75) - Récompense tirs proches
  - `ball_distance` (+0.02/step) - Guide vers ballon
  - `dribble_progress` (+0.05/step) - Progression avec ballon
- **Pénalités anti-CSC**: dangerous_shot, approach_own_goal
- **Pénalités hors limites**: out_of_bounds (-1.0)
- **Shaping annealing**: Réduit progressivement les dense rewards (1.0 → 0.1 sur 10M steps)

**Utilisé par**: `src/gfrl/env/rewarders.py`

---

### `curriculum_5v5.yaml`
**Rôle**: Définit les phases de curriculum learning

**Contenu**:
- Séquence de scénarios (3v1, 3v2, 3v3, 5v5)
- Critères de succès (winrate_min, min_episodes)
- Durée de chaque phase (duration_steps)
- Auto-advance settings

**Utilisé par**: `src/gfrl/train/trainer.py`

**Note**: Phase 0 (empty_goal) actuellement commentée, démarre directement en 3v1.

---

## 💻 Code Source

Fichiers Python situés dans `src/gfrl/`.

### Training Pipeline

#### `train/trainer.py`
**Rôle**: Orchestrateur principal de l'entraînement

**Responsabilités**:
- Initialise l'environnement et le modèle
- Gère le curriculum learning (transitions entre phases)
- Boucle d'entraînement principale:
  1. Collecte rollouts (256 steps × 24 envs)
  2. Calcule GAE returns
  3. Lance PPO update (4 epochs)
  4. Update RND (si activé)
  5. Logging et checkpointing
- Gère les LSTM states entre steps
- Reset LSTM states sur `done`

**Dépendances**: PPO, RolloutStorage, Policy, Environnements

---

#### `algo/ppo.py`
**Rôle**: Implémentation de l'algorithme PPO

**Responsabilités**:
- Calcul des losses:
  - **Policy loss**: Clipped surrogate objective
  - **Value loss**: Clipped MSE avec returns
  - **Entropy loss**: Bonus pour exploration
- Gradient clipping (max_grad_norm=0.5)
- AMP (Automatic Mixed Precision) support
- KL divergence early stopping
- Gestion des séquences LSTM (full batch mode)

**Fonction clé**: `update(storage)` - Entraîne la policy sur les rollouts collectés

---

#### `algo/storage.py`
**Rôle**: Buffer de rollouts pour PPO

**Responsabilités**:
- Stocke les transitions (obs, actions, rewards, values, lstm_states)
- Calcul GAE (Generalized Advantage Estimation):
  - Returns = advantages + values
  - Masking sur `done` pour reset
- Générateur de batches:
  - **Full batch mode** (mini_batch_size=None): Séquences (num_envs, num_steps)
  - **Mini-batch mode**: Random shuffling
- Normalisation des advantages

**Note**: Full batch obligatoire pour LSTM (préserve temporalité)

---

### Modèles

#### `models/policy_lstm.py`
**Rôle**: Architecture du réseau policy-value LSTM

**Architecture**:
```
Input (obs_dim=115)
  ↓
Backbone MLP (256 → 256 → 256) + LayerNorm
  ↓
LSTM (2 layers, 128 hidden)
  ↓
┌─────────────┬──────────────┐
Policy Head   Value Head
(19 actions)  (1 value)
```

**Spécificités**:
- **Policy head init**: `orthogonal_(gain=0.1)` - Crucial pour apprentissage
- **LSTM state management**: Reset sur épisodes terminés
- **Sequence handling**: Support batch (B, obs_dim) et séquence (B, T, obs_dim)

**Fonctions**:
- `forward()`: Policy logits + value
- `get_action()`: Sample action depuis policy
- `evaluate_actions()`: Log probs pour PPO update

---

### Environnement

#### `env/make_env.py`
**Rôle**: Factory pour créer les environnements GRF

**Responsabilités**:
- Crée l'env GRF de base (`football_env.create_environment`)
- Applique les wrappers dans l'ordre:
  1. `GRFtoGymnasiumWrapper` - Conversion Gym → Gymnasium
  2. `MirrorWrapper` - Data augmentation (optionnel)
  3. `RewardShaperWrapper` - Applique reward shaping
  4. `ObsWrapperRaw` - Encode observations raw en vecteur dense
  5. `RunningNormWrapper` - Normalise observations
- Support vectorisation (AsyncVectorEnv)

**Fonction clé**: `create_vec_envs()` - Crée 24 envs parallèles

---

#### `env/wrappers.py`
**Rôle**: Wrappers Gymnasium pour GRF

**Classes principales**:

1. **`GRFtoGymnasiumWrapper`**: Conversion Gym (old) → Gymnasium (new)
   - Adapte API step/reset
   
2. **`RewardShaperWrapper`**: Applique reward shaping
   - Multiplie goal rewards (×5 pour goals, ×2 pour wins)
   - Détecte CSC (own goals) et applique pénalité massive (-10.0)
   - Ajoute shaped rewards via `RewardShaper.compute_shaping()`
   - Ajoute win/draw/loss bonus en fin d'épisode

3. **`ObsWrapperRaw`**: Encode observations
   - Utilise `RawObsEncoder` pour vectoriser
   - Observations → vecteur dense (115 dims)

4. **`RunningNormWrapper`**: Normalisation running
   - Normalise observations avec mean/std cumulatifs

---

#### `env/rewarders.py`
**Rôle**: Calcul du reward shaping modulaire

**Classe**: `RewardShaper`

**Méthode principale**: `compute_shaping(prev_obs, obs, action, info)`

**Calcule**:
- **Shot rewards** (avec zone check x≥0.75)
- **Pass rewards** (bonus si sous pression)
- **Ball possession/recovery**
- **Dense shaping** (annealé progressivement):
  - `goal_distance`: Se rapprocher du but (clippé à x≤1.0)
  - `penalty_box_bonus`: Être dans la surface
  - `dribble_progress`: Progresser avec ballon (clippé à x≤1.0)
- **Pénalités anti-CSC** (jamais annealées):
  - Tir vers propre but
  - Passe dangereuse en défense
  - Approche propre but avec ballon
- **Out of bounds penalty**: Forte pénalité (-1.0) si x>1.0 ou hors terrain

**Annealing**: Multiplie shaped rewards par `annealing_multiplier` (1.0 → 0.1 sur 10M steps)

---

#### `env/encoders.py`
**Rôle**: Encode observations GRF raw en vecteurs

**Classe**: `RawObsEncoder`

**Encode** (115 dimensions):
- **Ball** (3): [x, y, z]
- **Ball direction** (3): [dx, dy, dz]
- **Ball owned** (3): [team, player_idx, one-hot]
- **Active player** (88): Position, direction, vitesse, etc. (8 dims)
- **Teammates** (10 × 8 = 80): Positions relatives, directions
- **Adversaires** (11 × 8 = 88): Idem
- **Game mode** (7): one-hot encoding

**Note**: Normalise positions pour être dans [-1, 1]

---

## 🚀 Scripts

Scripts d'exécution situés dans `src/gfrl/cli/`.

### `train_ppo_rnd.py`
**Rôle**: Point d'entrée pour l'entraînement

**Usage**:
```bash
python -m gfrl.cli.train_ppo_rnd --config configs/train_5v5.yaml
```

**Responsabilités**:
- Parse arguments CLI
- Charge configs (train, env, rewards, curriculum)
- Initialise `Trainer`
- Lance `trainer.train()`

---

### `scripts/watch_agent.py`
**Rôle**: Visualise un agent entraîné

**Usage**:
```bash
python scripts/watch_agent.py --checkpoint checkpoints/last.pt --render --env academy_3_vs_1_with_keeper --num-games 50
```

**Responsabilités**:
- Charge checkpoint
- Crée env avec rendering
- Roule agent en mode déterministe
- Affiche stats (goals, winrate)

---

## 🔄 Pipeline d'Exécution

### 1. Démarrage
```
train_ppo_rnd.py
  ├─ Load configs (train_5v5.yaml, env.yaml, rewards_dense.yaml, curriculum_5v5.yaml)
  ├─ Create Trainer
  └─ trainer.train()
```

### 2. Initialisation Trainer
```
Trainer.__init__()
  ├─ Create RewardShaper (rewards_dense.yaml)
  ├─ Create vectorized envs (24 parallel)
  │   └─ Apply wrappers (RewardShaper, ObsEncoder, RunningNorm)
  ├─ Create PolicyLSTM (obs_dim=115, action_dim=19)
  ├─ Create PPO optimizer
  ├─ Create RolloutStorage (num_steps=256, num_envs=24)
  └─ Load curriculum phases
```

### 3. Boucle d'Entraînement
```
For each update (8138 updates = 50M steps):
  
  # 1. Collect rollouts (256 steps × 24 envs = 6144 samples)
  For step in range(256):
    ├─ obs → policy.get_action() → action, log_prob, value, lstm_state
    ├─ envs.step(action) → next_obs, reward, done, info
    ├─ storage.insert(obs, action, log_prob, value, reward, done, lstm_state)
    └─ If done: Reset LSTM state for that env
  
  # 2. Compute GAE returns
  ├─ next_value = policy.get_value(last_obs)
  └─ storage.compute_returns(next_value, gamma=0.993, gae_lambda=0.95)
  
  # 3. PPO update (4 epochs, full batch pour LSTM)
  ├─ For epoch in range(4):
  │   └─ batch = storage.get_generator(batch_size=6144, mini_batch_size=None)
  │       ├─ Reshape to sequences: (num_envs=24, num_steps=256, obs_dim)
  │       ├─ policy.evaluate_actions(obs_seq, actions_seq, lstm_state)
  │       ├─ Compute losses (policy + value + entropy)
  │       ├─ Backward + gradient clip
  │       └─ Optimizer step
  
  # 4. Update RND (si enabled)
  └─ rnd.update() - Curiosité intrinsèque
  
  # 5. Logging & Checkpointing
  ├─ Log metrics (rewards, entropy, losses, KL)
  ├─ Check curriculum advancement
  └─ Save checkpoint every 50 updates
```

### 4. Reward Computation (par step)
```
env.step(action)
  ├─ base_reward = GRF sparse reward (±1 pour goals)
  │
  ├─ RewardShaperWrapper:
  │   ├─ Multiply goal rewards: base_reward × goal_scored (×5)
  │   ├─ Detect CSC: Si goal concédé en academy no-opponent → -10.0
  │   │
  │   ├─ RewardShaper.compute_shaping():
  │   │   ├─ Shot rewards (+1.5 si x≥0.75, pénalité sinon)
  │   │   ├─ Pass rewards (+0.2, +0.1 si sous pression)
  │   │   ├─ Dense shaping (annealé):
  │   │   │   ├─ goal_distance: +0.08 × progress (clippé x≤1.0)
  │   │   │   ├─ penalty_box_bonus: +0.15 si x≥0.75
  │   │   │   └─ dribble_progress: +0.05 × progress (clippé x≤1.0)
  │   │   ├─ Anti-CSC penalties (non-annealé)
  │   │   └─ Out of bounds: -1.0 si x>1.0 ou hors terrain
  │   │
  │   ├─ shaped_reward *= annealing_multiplier
  │   ├─ total_reward = base_reward + shaped_reward
  │   │
  │   └─ If episode done: total_reward += win/draw/loss bonus
  │
  └─ Return (obs, total_reward, done, info)
```

---

## 🐛 Bugs Corrigés

### Bug #1: Policy head gain trop faible
- **Avant**: `gain=0.01` → logits quasi-nuls → distribution uniforme
- **Après**: `gain=0.1` → apprentissage possible
- **Fichier**: `src/gfrl/models/policy_lstm.py`

### Bug #2: Mini-batches cassent LSTM
- **Avant**: Mini-batches avec LSTM state=0 → casse temporalité
- **Après**: Full batch mode (mini_batch_size=None)
- **Fichier**: `src/gfrl/train/trainer.py`

### Bug #3: Shape mismatch LSTM
- **Avant**: Flatten (num_steps × num_envs) + LSTM state (num_envs) → crash
- **Après**: Reshape en séquences (num_envs, num_steps, obs_dim)
- **Fichier**: `src/gfrl/algo/storage.py`

### Bug #4: Goal rewards non appliqués
- **Avant**: goal_scored config ignoré, goals valent +1
- **Après**: Multiplié par config (×5)
- **Fichier**: `src/gfrl/env/wrappers.py`

### Bug #5: Agent court derrière le but
- **Avant**: `abs(x - 1.0)` → x=1.1 a même distance que x=0.9
- **Après**: Clip à x≤1.0 + pénalité out_of_bounds
- **Fichier**: `src/gfrl/env/rewarders.py`

---

## 📊 Configuration Actuelle

**Hyperparamètres PPO**:
- Learning rate: 3e-4
- Gamma: 0.993
- GAE lambda: 0.95
- Clip epsilon: 0.2
- Entropy: 0.01 → 0.001 (5M steps)
- Batch: 6144 (full batch pour LSTM)
- Epochs: 4

**Architecture**:
- Backbone: MLP 256×3 + LayerNorm
- LSTM: 2 layers, 128 hidden
- Policy head gain: 0.1 ✅

**Rewards** (config équilibrée post-bugfix):
- goal_scored: +5.0
- win_bonus: +2.0
- shot_attempt: +1.5 (si x≥0.75)
- penalty_box_bonus: +0.15/step (si x≥0.75)
- goal_distance: +0.08/step
- own_goal: -10.0
- out_of_bounds: -1.0

**Environnement**:
- 24 envs parallèles
- Rollout: 256 steps
- Représentation: raw (115 dims)
- Action set: sticky (19 actions)

---

## 🎯 Fichiers NON Utilisés

Ces fichiers existent mais ne sont **pas** dans le pipeline actuel:

- `src/gfrl/algo/rnd.py` - RND désactivé (use_rnd: false)
- `src/gfrl/models/cnn_policy.py` - On utilise raw obs, pas pixels
- `scripts/eval_agent.py` - Évaluation formelle non utilisée
- Tout autre fichier non mentionné ci-dessus

---

**Date**: 20 octobre 2025  
**Version**: Post-bugfix (5 bugs critiques corrigés)  
**Status**: Prêt pour entraînement sur 3v1 → 3v2 → 3v3 → 5v5
