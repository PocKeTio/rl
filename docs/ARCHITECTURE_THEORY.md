# Architecture & Théorie du Projet - Deep Reinforcement Learning pour Football

## 📚 Table des matières
1. [Agents (PPO + Policy)](#agents)
2. [Environnements (Wrappers & Rewards)](#environnements)
3. [Modèles (Réseaux de neurones)](#modèles)
4. [Training (Entraînement)](#training)
5. [Evaluation (Métriques)](#evaluation)

---

## 1. Agents (PPO + Policy) 🤖

### `agents/ppo.py` - Proximal Policy Optimization

#### **Théorie: Qu'est-ce que PPO?**

PPO est un algorithme d'apprentissage par renforcement **on-policy** qui optimise une politique (policy) en maximisant les récompenses tout en restant proche de l'ancienne politique.

**Problème résolu:**
- Les algorithmes policy gradient classiques (REINFORCE) sont instables: un mauvais gradient peut détruire la politique
- PPO limite les changements de politique à chaque update pour garantir la stabilité

**Équation clé - Objectif PPO:**
```
L^CLIP(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

Où:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (ratio de probabilités)
- A_t = advantage (à quel point l'action était meilleure que la moyenne)
- ε = clip_epsilon (typiquement 0.2)
```

**Intuition:**
- Si l'action était bonne (A_t > 0), on augmente sa probabilité
- Mais on limite l'augmentation avec le clipping pour éviter les changements trop brutaux
- Si l'action était mauvaise (A_t < 0), on diminue sa probabilité (avec clipping aussi)

**Composants dans le code:**

1. **Policy Loss** (ligne ~140):
   ```python
   ratio = exp(new_log_prob - old_log_prob)  # Ratio de probabilités
   surr1 = ratio * advantages                 # Objectif non clippé
   surr2 = clip(ratio, 1-ε, 1+ε) * advantages # Objectif clippé
   policy_loss = -min(surr1, surr2)           # On prend le min (pessimiste)
   ```

2. **Value Loss** (ligne ~143):
   ```python
   value_loss = (returns - values)^2  # Erreur quadratique
   ```
   - Le critique (value function) estime V(s) = récompense future attendue
   - On minimise l'erreur entre la prédiction et le return réel

3. **Entropy Bonus** (ligne ~152):
   ```python
   entropy_loss = -entropy  # On maximise l'entropie
   ```
   - Encourage l'exploration en pénalisant les politiques trop déterministes
   - Diminue progressivement (annealing) pour converger vers une politique déterministe

**Hyperparamètres importants:**
- `clip_epsilon = 0.2`: Limite les changements de politique (±20%)
- `value_coef = 0.5`: Poids de la value loss
- `entropy_coef = 0.01→0.002`: Poids de l'entropie (décroît avec le temps)
- `ppo_epochs = 4`: Nombre de passes sur les données collectées

---

### `agents/policy.py` - Réseau de politique LSTM

#### **Théorie: Pourquoi LSTM?**

Le football nécessite de la **mémoire temporelle**:
- Se souvenir où sont les joueurs
- Anticiper les trajectoires
- Planifier des séquences d'actions (dribble, passe, tir)

**Architecture:**
```
Observation (161 dim)
    ↓
MLP Encoder (512 hidden units)
    ↓
LSTM (128 hidden, 1 layer)
    ↓
    ├→ Actor Head → Distribution d'actions (19 actions)
    └→ Critic Head → Value V(s)
```

**Composants:**

1. **Encoder MLP** (ligne ~50):
   ```python
   obs → Linear(512) → ReLU → Linear(512) → ReLU
   ```
   - Extrait des features de l'observation brute
   - 512 dimensions pour capturer la complexité du jeu

2. **LSTM** (ligne ~60):
   ```python
   features → LSTM(128) → hidden_state
   ```
   - **Hidden state (h_t)**: Mémoire à court terme (position actuelle)
   - **Cell state (c_t)**: Mémoire à long terme (contexte du match)
   - Permet de se souvenir des 10-20 derniers steps

3. **Actor (Policy)** (ligne ~70):
   ```python
   hidden → Linear(action_dim) → Categorical distribution
   ```
   - Sortie: Probabilités pour chaque action (19 actions possibles)
   - Exemple: [0.1, 0.05, 0.3, ...] → 30% de chance de tirer

4. **Critic (Value)** (ligne ~80):
   ```python
   hidden → Linear(1) → V(s)
   ```
   - Sortie: Valeur de l'état (récompense future attendue)
   - Utilisé pour calculer les advantages

**Pourquoi séparer Actor et Critic?**
- **Actor-Critic architecture**: Deux têtes qui partagent le même encoder
- Actor apprend QUOI faire (actions)
- Critic apprend COMBIEN ça vaut (valeur)
- Plus stable que d'avoir un seul réseau

---

## 2. Environnements (Wrappers & Rewards) 🏟️

### `envs/wrappers.py` - Wrappers d'environnement

#### **Théorie: Pourquoi des wrappers?**

Google Research Football (GRF) retourne des observations brutes complexes. Les wrappers:
1. Simplifient les observations
2. Normalisent les données
3. Ajoutent du reward shaping

**Wrappers principaux:**

#### 1. **GRFtoGymnasiumWrapper** (ligne ~50)
```python
Objectif: Convertir GRF (ancien Gym) vers Gymnasium (nouveau standard)
```

**Problème résolu:**
- GRF utilise l'ancienne API Gym (step retourne 4 valeurs)
- Gymnasium utilise la nouvelle API (step retourne 5 valeurs)
- Conversion: `(obs, reward, done, info)` → `(obs, reward, terminated, truncated, info)`

**Fonctions clés:**
- `step()`: Convertit le format de sortie
- `get_original_obs()`: Récupère l'observation brute (pour stats)
- `get_current_score()`: Extrait le score du match

#### 2. **ObsWrapperRaw** (ligne ~150)
```python
Objectif: Encoder les observations brutes en vecteur numérique
```

**Transformation:**
```
Observation GRF (dict avec ~30 clés)
    ↓
Vecteur numérique (161 dimensions)
```

**Exemple d'encodage:**
- Positions des joueurs: `left_team` (11 joueurs × 2 coords = 22 dims)
- Position du ballon: `ball` (3 coords)
- Direction du ballon: `ball_direction` (3 coords)
- Joueur actif: `active` (1 dim)
- Sticky actions: `sticky_actions` (10 dims)
- Score: `score` (2 dims)
- etc.

**Pourquoi 161 dimensions?**
- Somme de toutes les features pertinentes
- Assez compact pour le réseau de neurones
- Contient toute l'information nécessaire

#### 3. **RunningNormWrapper** (ligne ~250)
```python
Objectif: Normaliser les observations pour stabiliser l'apprentissage
```

**Théorie: Pourquoi normaliser?**
- Les réseaux de neurones apprennent mieux avec des données normalisées (moyenne=0, std=1)
- Sans normalisation: certaines features dominent (ex: positions en [-1, 1] vs vitesses en [-0.1, 0.1])

**Méthode:**
```python
obs_normalized = (obs - running_mean) / (running_std + epsilon)
```

**Running statistics:**
- Moyenne et écart-type calculés **en ligne** (pendant le training)
- Mis à jour à chaque step avec une moyenne mobile
- Sauvegardés dans les checkpoints pour l'évaluation

**⚠️ Attention:**
- Désactivé par défaut (`running_norm: false`) car peut causer des incohérences train/eval
- Si activé, il faut utiliser les mêmes stats en eval

---

### `envs/make_env.py` - Création d'environnements

#### **Théorie: Environnements vectorisés**

**Problème:**
- Collecter des données avec 1 seul env est lent
- Solution: Paralléliser avec N environnements

**AsyncVectorEnv:**
```python
24 environnements en parallèle
    ↓
24 workers (processus séparés)
    ↓
Collecte 24× plus de données par seconde
```

**Avantages:**
1. **Vitesse**: 24× plus rapide
2. **Diversité**: Expériences variées (différents états de jeu)
3. **Stabilité**: Moyennes sur plusieurs envs → gradients plus stables

**Fonctionnement:**
```python
obs = envs.reset()  # Reset 24 envs → (24, 161)
actions = policy(obs)  # 24 actions
next_obs, rewards, dones = envs.step(actions)  # Step parallèle
```

---

### `envs/rewards.py` - Reward Shaping

#### **Théorie: Pourquoi du reward shaping?**

**Problème des récompenses sparses:**
- GRF donne +1 pour un but, -1 pour un but encaissé
- Un but arrive tous les ~100-500 steps
- L'agent ne reçoit presque jamais de feedback → n'apprend rien

**Solution: Reward Shaping**
```
Reward total = Reward GRF (sparse) + Reward shaping (dense)
```

**Récompenses denses (rewards_dense.yaml):**

1. **Ball distance** (+0.005):
   ```python
   reward = -distance_to_ball * 0.005
   ```
   - Encourage l'agent à aller vers le ballon
   - Feedback constant

2. **Goal distance** (+0.02):
   ```python
   reward = -(distance_to_goal - prev_distance_to_goal) * 0.02
   ```
   - Récompense pour se rapprocher du but adverse
   - Pénalité pour s'en éloigner

3. **Shot attempt** (+5.0):
   ```python
   if action == SHOT and ball_x > 0.85 and ball_direction_x > 0:
       reward = +5.0
   ```
   - ÉNORME récompense pour tirer (proche du but ET vers le but)
   - Force l'exploration du tir

4. **Pass** (+0.1):
   ```python
   if ball_owner_changed and same_team and forward_pass:
       reward = +0.1
   ```
   - Encourage les passes vers l'avant

**Annealing (décroissance):**
```python
reward_shaped = reward_shaped * annealing_factor
annealing_factor: 1.0 → 0.0 (sur 10M steps)
```
- Au début: Beaucoup de shaping (guide l'agent)
- À la fin: Seulement les vrais buts comptent (performance réelle)

---

## 3. Modèles (Réseaux de neurones) 🧠

### `models/policy_lstm.py` - Architecture de la politique

Voir section [Agents/Policy](#agents-policy---réseau-de-politique-lstm) ci-dessus.

---

### `models/encoders.py` - Encodeurs d'observations

#### **Théorie: Pourquoi un encodeur?**

**Problème:**
- Observation GRF = dict complexe avec ~30 clés
- Réseau de neurones = besoin d'un vecteur numérique fixe

**Solution: Encoder**
```
Dict GRF → Vecteur (161 dims) → Réseau de neurones
```

**Types d'encodeurs:**

#### 1. **RawObsEncoder** (Simple115 style)
```python
Concatène toutes les features importantes:
- Positions des joueurs (22 dims)
- Position du ballon (3 dims)
- Vitesses (22 dims)
- Sticky actions (10 dims)
- Game mode (7 dims one-hot)
- etc.
Total: 161 dimensions
```

**Avantages:**
- Simple et efficace
- Toute l'information est préservée
- Fonctionne bien avec LSTM

#### 2. **SMM Encoder** (Spatial Mini-Map)
```python
Convertit les positions en image 2D:
- Terrain divisé en grille (ex: 96×72)
- Chaque joueur = pixel
- Canaux: [joueurs alliés, joueurs adverses, ballon]
```

**Avantages:**
- Représentation spatiale naturelle
- Peut utiliser des CNN
- Bon pour la vision globale du jeu

**Dans ce projet:**
- On utilise **RawObsEncoder** (plus simple et efficace)
- SMM est plus adapté pour des approches visuelles

---

## 4. Training (Entraînement) 🎓

### `training/trainer.py` - Boucle d'entraînement principale

#### **Théorie: Comment entraîner un agent RL?**

**Algorithme général (PPO):**
```
1. Collecter des expériences (rollouts)
2. Calculer les advantages
3. Optimiser la politique avec PPO
4. Répéter
```

**Détails de chaque étape:**

#### **Étape 1: Collecte de rollouts** (ligne ~450)
```python
for step in range(rollout_len):  # 256 steps
    action = policy(obs, lstm_state)
    next_obs, reward, done = env.step(action)
    storage.add(obs, action, reward, value, log_prob)
```

**Qu'est-ce qu'un rollout?**
- Séquence d'expériences: (s_0, a_0, r_0, s_1, a_1, r_1, ...)
- Longueur: 256 steps
- 24 environnements en parallèle → 6144 transitions par rollout

**Pourquoi 256 steps?**
- Compromis entre:
  - Court: Updates fréquents (plus stable)
  - Long: Plus de contexte pour LSTM (meilleure mémoire)
- 256 = ~10-20 secondes de jeu

#### **Étape 2: Calcul des advantages** (ligne ~620)
```python
advantages = GAE(rewards, values, next_value)
```

**Théorie: Generalized Advantage Estimation (GAE)**

**Problème:**
- Comment savoir si une action était bonne?
- Comparer avec quoi?

**Solution: Advantage**
```
A(s,a) = Q(s,a) - V(s)
       = "Valeur de l'action" - "Valeur moyenne de l'état"
```

**Intuition:**
- A > 0: Action meilleure que la moyenne → augmenter sa probabilité
- A < 0: Action pire que la moyenne → diminuer sa probabilité

**GAE Formula:**
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

Où:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
- γ = 0.993 (discount factor)
- λ = 0.95 (GAE parameter)
```

**Pourquoi GAE?**
- Compromis entre:
  - λ=0: Faible variance, biais élevé (TD)
  - λ=1: Variance élevée, pas de biais (Monte Carlo)
- λ=0.95: Bon équilibre

#### **Étape 3: Optimisation PPO** (ligne ~630)
```python
for epoch in range(4):  # 4 epochs
    for minibatch in get_batches(data, minibatch_size=3072):
        loss = ppo_loss(minibatch)
        loss.backward()
        optimizer.step()
```

**Détails:**
- **4 epochs**: On passe 4 fois sur les mêmes données
- **Mini-batches**: Divise les 6144 transitions en batches de 3072
- **Gradient clipping**: Limite les gradients à 0.5 (stabilité)

**Pourquoi plusieurs epochs?**
- Réutiliser les données collectées (sample efficiency)
- Mais pas trop (sinon overfitting sur les vieilles données)

---

### `training/storage.py` - Rollout Buffer

#### **Théorie: Pourquoi un buffer?**

**Problème:**
- On collecte 6144 transitions
- On doit les stocker quelque part
- On doit pouvoir les échantillonner en mini-batches

**Solution: RolloutStorage**

**Structure:**
```python
storage = {
    'obs': Tensor(256, 24, 161),      # Observations
    'actions': Tensor(256, 24),        # Actions
    'rewards': Tensor(256, 24),        # Rewards
    'values': Tensor(256, 24),         # Values V(s)
    'log_probs': Tensor(256, 24),      # Log probabilities
    'lstm_h': Tensor(256, 24, 128),    # LSTM hidden states
    'lstm_c': Tensor(256, 24, 128),    # LSTM cell states
    'returns': Tensor(256, 24),        # Computed returns
    'advantages': Tensor(256, 24),     # Computed advantages
}
```

**Fonctions clés:**

1. **add()** (ligne ~100):
   ```python
   storage.add(step, obs, action, reward, value, log_prob, lstm_state)
   ```
   - Ajoute une transition au buffer
   - Appelé à chaque step de la collecte

2. **compute_returns()** (ligne ~150):
   ```python
   returns, advantages = compute_gae(rewards, values, next_value)
   ```
   - Calcule les returns et advantages avec GAE
   - Appelé après la collecte, avant l'optimisation

3. **get_generator()** (ligne ~200):
   ```python
   for batch in storage.get_generator(batch_size, minibatch_size):
       yield batch
   ```
   - Génère des mini-batches pour l'optimisation
   - Shuffle les données pour éviter l'overfitting

**Pourquoi stocker les LSTM states?**
- PPO a besoin de réévaluer les actions avec la nouvelle politique
- Il faut le même contexte LSTM que lors de la collecte
- Sinon les probabilités seraient fausses

---

### `training/curriculum.py` - Curriculum Learning

#### **Théorie: Qu'est-ce que le curriculum learning?**

**Problème:**
- Apprendre 11v11 directement est trop difficile
- L'agent ne marque jamais de but → n'apprend rien

**Solution: Curriculum**
```
Facile → Moyen → Difficile
```

**Exemple de curriculum:**
```
Phase 1: academy_empty_goal_close (10M steps)
    ↓ (si winrate > 80%)
Phase 2: academy_empty_goal (10M steps)
    ↓ (si winrate > 70%)
Phase 3: academy_run_to_score (15M steps)
    ↓ (si winrate > 60%)
Phase 4: academy_3_vs_1_with_keeper (20M steps)
    ↓ (si winrate > 40%)
Phase 5: 11_vs_11_easy_stochastic (50M steps)
```

**Critères d'avancement:**
- **Winrate**: % de victoires sur les 100 derniers épisodes
- **Min steps**: Nombre minimum de steps avant de pouvoir avancer
- **Max steps**: Nombre maximum de steps (force l'avancement)

**Avantages:**
1. **Apprentissage progressif**: Maîtrise les bases avant les situations complexes
2. **Sample efficiency**: Apprend plus vite qu'en 11v11 direct
3. **Transfert**: Les compétences apprises se transfèrent aux phases suivantes

**Implémentation:**
```python
if should_advance_curriculum():
    # Fermer les anciens envs
    envs.close()
    
    # Créer les nouveaux envs (phase suivante)
    envs = create_vec_envs(new_env_name)
    
    # Reset LSTM states
    lstm_state = reset_lstm()
    
    # Continuer l'entraînement
```

---

## 5. Evaluation (Métriques) 📊

### `evaluation/evaluator.py` - Évaluation de l'agent

#### **Théorie: Pourquoi évaluer?**

**Problème:**
- Pendant le training, l'agent explore (actions aléatoires)
- On veut savoir sa vraie performance (sans exploration)

**Solution: Évaluation périodique**
```python
Tous les 25000 steps:
    1. Freeze la politique (mode déterministe)
    2. Jouer N épisodes (ex: 10)
    3. Calculer les métriques (winrate, goals, etc.)
    4. Logger dans TensorBoard
```

**Mode déterministe:**
```python
# Training (stochastique)
action = sample(policy(obs))  # Tire une action selon les probas

# Eval (déterministe)
action = argmax(policy(obs))  # Prend l'action la plus probable
```

**Métriques évaluées:**
- **Winrate**: % de victoires
- **Goals scored/conceded**: Buts marqués/encaissés
- **Episode length**: Durée moyenne des matchs
- **Return**: Récompense totale moyenne

---

### `evaluation/metrics.py` - Football Stats Tracker

#### **Théorie: Métriques de football avancées**

**Objectif:**
- Comprendre COMMENT l'agent joue
- Pas seulement s'il gagne, mais COMMENT il gagne

**Métriques trackées:**

#### 1. **Possession** (% du temps avec le ballon)
```python
possession = steps_with_ball / total_steps * 100
```
- Indicateur de domination
- Bon agent: 60-70% de possession

#### 2. **Tirs** (shots per game)
```python
shots = count(action == SHOT)
shots_on_target = count(action == SHOT and ball_direction_x > 0)
shot_accuracy = shots_on_target / shots * 100
```
- Indicateur d'agressivité
- Bon agent: 5-10 tirs/match, 50-70% cadrés

#### 3. **Passes** (passes per game)
```python
passes = count(ball_owner_changed and same_team)
pass_accuracy = passes_completed / passes_attempted * 100
```
- Indicateur de jeu collectif
- Bon agent: 70-80% de passes réussies

#### 4. **Zones de jeu** (% du temps dans chaque zone)
```python
if x < -0.33: defense
elif x < 0.33: midfield
else: attack
```
- Indicateur de positionnement
- Bon agent: 20% défense, 30% milieu, 50% attaque

#### 5. **Heatmaps** (positions cumulées)
```python
heatmap[x, y] += 1  # À chaque step
```
- Visualisation des zones fréquentées
- Permet de voir les patterns de jeu

**Logging dans TensorBoard:**
```python
writer.add_scalar("football/possession_pct", possession, step)
writer.add_scalar("football/shots_per_game", shots, step)
writer.add_image("heatmap/player", heatmap, step)
```

---

## 🎯 Résumé: Comment tout fonctionne ensemble

```
1. ENVIRONNEMENT (envs/)
   ↓
   Observations brutes (dict)
   ↓
2. ENCODEUR (models/encoders.py)
   ↓
   Vecteur numérique (161 dims)
   ↓
3. POLITIQUE (models/policy_lstm.py)
   ↓
   Action + Value + LSTM state
   ↓
4. ENVIRONNEMENT
   ↓
   Reward + Next obs
   ↓
5. STORAGE (training/storage.py)
   ↓
   Buffer de 6144 transitions
   ↓
6. GAE (training/trainer.py)
   ↓
   Advantages calculés
   ↓
7. PPO (agents/ppo.py)
   ↓
   Optimisation de la politique
   ↓
8. REPEAT (avec curriculum + eval périodique)
```

---

## 📚 Références théoriques

### Papers fondamentaux:
1. **PPO**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
2. **GAE**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
3. **LSTM**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
4. **Curriculum Learning**: "Curriculum Learning" (Bengio et al., 2009)

### Concepts clés:
- **On-policy**: L'agent apprend de sa propre politique actuelle
- **Actor-Critic**: Deux réseaux (policy + value) qui s'entraident
- **Advantage**: Mesure de la qualité d'une action par rapport à la moyenne
- **Clipping**: Limite les changements de politique pour la stabilité
- **Entropy**: Mesure de l'aléatoire (exploration vs exploitation)

---

## 💡 Conseils pour comprendre le code

1. **Commencez par les concepts:**
   - Lisez d'abord cette doc
   - Regardez des vidéos sur PPO (ex: Arxiv Insights)
   - Comprenez le problème avant la solution

2. **Suivez le flow:**
   - Partez de `scripts/train.py`
   - Suivez l'exécution step by step
   - Utilisez un debugger pour voir les tensors

3. **Expérimentez:**
   - Changez les hyperparamètres
   - Observez l'impact sur l'apprentissage
   - Utilisez TensorBoard pour visualiser

4. **Lisez les papers:**
   - PPO paper (très accessible)
   - GAE paper (pour les advantages)
   - Google Research Football paper (pour l'env)

---

**Auteur:** Documentation générée pour le projet Football RL
**Date:** 2025
**Version:** 1.0
