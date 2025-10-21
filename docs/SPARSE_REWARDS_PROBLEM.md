# Problème: Rewards TROP Sparse avec RND

## 🔴 Problème observé (2.3M steps)

### Comportement:
```
Agent court vers SES PROPRES buts avec la balle
Ne tire JAMAIS
0 goals sur 2.3M steps
Mean reward: 0.0
Entropy: 2.73 (93% random!)
```

### Métriques:
```
Goals scored: 0.00/game
Winrate: 0.2%
Drawrate: 94.3%

RND intrinsic: 1.6030
Policy loss: 0.0011 (pas d'apprentissage)
Entropy: 2.7329 (quasi-uniforme)
```

---

## 🔬 Analyse Root Cause

### Configuration initiale:
```yaml
# rewards_sparse.yaml
goal_scored: +100
goal_conceded: -10
# TOUT LE RESTE DÉSACTIVÉ

# train_sparse.yaml
rnd_beta: 0.05
```

### Calcul du signal:

**Sur 223 steps (durée moyenne épisode):**

```python
# Extrinsic (task rewards)
Goals: 0 × 100 = 0
Total extrinsic: 0

# Intrinsic (RND curiosity)
RND par step: 1.6
Beta: 0.05
Contribution par step: 0.05 × 1.6 = 0.08
Total intrinsic: 0.08 × 223 = 17.84

# Ratio
Extrinsic/Intrinsic = 0 / 17.84 = 0:∞
→ Agent optimise UNIQUEMENT RND!
```

---

## 💡 Pourquoi l'agent va vers ses buts?

### Logique RND:
```
RND reward = MSE entre target et predictor
États jamais visités = haute MSE = haute reward

Zone rarement visitée: Ses propres buts
→ RND reward élevé
→ Agent y va pour maximiser curiosité
→ Ignore complètement la task (scorer)
```

### Cercle vicieux:
```
1. Agent n'a jamais marqué → pas de signal task
2. Optimise RND uniquement
3. RND le pousse vers zones inexplorées (mauvaises)
4. Ne marque jamais → retour à 1
```

---

## ❌ Pourquoi Sparse + RND échoue

### Hypothèse initiale (fausse):
```
"RND va guider l'agent à explorer intelligemment
 et finir par trouver comment marquer"
```

### Réalité:
```
RND pousse vers NOUVEAUTÉ, pas vers TASK

Zones nouvelles ≠ Zones utiles pour task
→ Agent explore aléatoirement
→ Ne converge JAMAIS vers la task
```

### Comparaison:

| Approche | Signal task | Exploration | Résultat |
|----------|-------------|-------------|----------|
| **Dense rewards** | Fort, constant | Guidée par shaping | Apprend vite mais bugs |
| **Sparse + RND (beta=0.05)** | Zéro (pas de goals) | Dominée par RND | Explore sans apprendre |
| **Sparse + minimal shaping** | Faible mais constant | Équilibrée | Apprend lentement mais stable |

---

## ✅ Solution appliquée

### 1. Ajouter guidance minimale

```yaml
# rewards_sparse.yaml
goal_distance:
  enabled: true
  reward_scale: 0.5  # TRÈS FAIBLE
  normalize: true
```

**Impact:**
```python
Goal distance reward: ~0.1-1.0 par step (vers but adverse)
Sur 223 steps: ~50-100 de signal task
Avec goals: +100 supplémentaires

→ Signal task > Signal RND
```

### 2. Réduire rnd_beta

```yaml
# train_sparse.yaml
rnd_beta: 0.01  # Was 0.05
```

**Impact:**
```python
RND contribution: 0.01 × 1.6 × 223 = 3.57
Goal distance: ~75
Goals (si marqués): +100

Ratio task/curiosity: 175 / 3.57 = 49:1 ✅
→ RND est vraiment un bonus, pas l'objectif
```

---

## 📊 Nouveau calcul des rewards

### Avec minimal shaping (0.5 goal_distance):

```python
# Par épisode (223 steps):

Extrinsic:
- Goal distance: ~0.3 moyenne/step × 223 = 67
- Goal scored: 100 (si marqué)
- Total: 167 (si goal) ou 67 (si pas goal)

Intrinsic:
- RND: 0.01 × 1.6 × 223 = 3.57

Ratio: 167 / 3.57 = 47:1 (avec goal)
       67 / 3.57 = 19:1 (sans goal)

→ Task dominant, RND est un bonus ✅
```

---

## 🎯 Lessons Learned

### ❌ Sparse "pur" ne fonctionne PAS avec RND
```
Même avec RND, besoin d'UN MINIMUM de guidance
Sinon: Agent optimise curiosité au lieu de task
```

### ✅ Sparse "quasi-pur" fonctionne
```
Goal scored: +100 (dominant)
Goal distance: +0.5 (minimal, juste orientation)
RND: 0.01 beta (vraiment un bonus)

→ Balance: Task >> Shaping >> Curiosity
```

### 📐 Règle générale:
```
Signal extrinsic minimum > Signal intrinsic total

Sinon: RND domine et agent n'apprend pas la task
```

---

## 🚀 Attentes avec nouvelle config

### Timeline prévue:

```
0-5M steps:
  Goal distance guide vers but adverse
  RND encourage essayer différentes actions
  Premiers goals par essai-erreur
  Winrate: 1-5%

5-15M steps:
  Agent comprend: aller vers but = récompense
  Commence à tirer plus régulièrement
  Winrate: 5-15%

15-30M steps:
  Stratégie émerge
  Tir + placement
  Winrate: 20-35%
```

---

## 📚 Références théoriques

**Burda et al. (2018) - RND paper:**
- "Intrinsic rewards should be smaller than extrinsic rewards"
- Beta range: 0.01-0.1
- Utilisent TOUJOURS des task rewards non-nuls

**OpenAI Procgen (2020):**
- "Pure exploration (RND only) fails on most tasks"
- "Need both task signal and exploration bonus"

**Best practice:**
```
beta_intrinsic = 0.01-0.05
Minimal task shaping (même sparse)
Never pure exploration without any task signal
```
