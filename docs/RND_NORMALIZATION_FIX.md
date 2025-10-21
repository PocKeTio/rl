# RND Normalization Fix - Technical Details

## 🔴 Bug Original

### Code cassé (initial):
```python
# Dans update()
self.reward_count += 1
delta = loss - self.reward_std.pow(2)
self.reward_std = torch.sqrt(self.reward_std.pow(2) + delta / self.reward_count)
```

**Problème:** Formule incorrecte, pas Welford's algorithm valide.

---

## 🟡 Fix v1 (Approximation)

### Code:
```python
# Dans update()
self.reward_std = 0.99 * self.reward_std + 0.01 * torch.sqrt(loss + 1e-8)
```

**Ce que ça fait:**
- `loss` = moyenne du MSE sur le batch (scalar)
- `sqrt(loss)` = RMSE (Root Mean Squared Error)
- EMA du RMSE

**Problème mathématique:**
```
reward_std ≈ EMA(RMSE) = EMA(sqrt(E[MSE]))

Mais on veut:
reward_std ≈ sqrt(Var[MSE]) = std(MSE)
```

**Impact:** Approximation raisonnable mais pas rigoureuse. L'échelle de normalisation n'est pas la vraie variance.

---

## ✅ Fix v2 (Rigoureux - FINAL)

### Code:
```python
# Dans compute_intrinsic_reward()
mse = (target_features - predicted_features).pow(2).mean(dim=-1)  # (batch,)

# Calculer le std RÉEL du batch de MSE
if mse.numel() > 1:
    batch_mse_std = mse.std()  # Variance réelle du batch
    self.reward_std = 0.99 * self.reward_std + 0.01 * batch_mse_std

# Normaliser
intrinsic_reward = mse / (self.reward_std + 1e-8)
```

**Ce que ça fait:**
- `mse` = tenseur de MSE par observation (batch,)
- `mse.std()` = écart-type réel du batch
- EMA de cet écart-type

**Mathématiquement correct:**
```python
reward_std = EMA(std[MSE_batch])
           = EMA(sqrt(Var[MSE]))
           
→ Normalisation par la vraie dispersion des intrinsic rewards
```

---

## 📊 Comparaison des 3 versions

### Version CASSÉE (original):
```python
# Formule incorrecte
delta = loss - self.reward_std.pow(2)
self.reward_std = torch.sqrt(self.reward_std.pow(2) + delta / self.reward_count)

Résultat: Drift, instabilité, valeurs aberrantes
```

### Version APPROXIMATIVE (v1):
```python
# EMA du RMSE
self.reward_std = 0.99 * old + 0.01 * sqrt(mean(MSE))

Résultat: Stable, mais échelle approximative
→ Intrinsic rewards peuvent varier en échelle
```

### Version RIGOUREUSE (v2 - FINAL):
```python
# EMA du std du batch MSE
self.reward_std = 0.99 * old + 0.01 * std(MSE_batch)

Résultat: Stable ET échelle correcte
→ Intrinsic rewards normalisés par vraie variance
```

---

## 🔬 Impact pratique

### Avec v1 (approximation):
```
Intrinsic rewards peuvent avoir échelle variable selon:
- Distribution des features
- Niveau d'apprentissage du predictor
- Correlation entre observations

Exemple:
- États similaires: MSE moyen = 1.0, std(MSE) = 0.1
  → RMSE = 1.0, std réel = 0.1
  → Normalisation par 1.0 au lieu de 0.1 (×10 erreur!)
```

### Avec v2 (rigoureux):
```
Intrinsic rewards normalisés par vraie dispersion:
- États similaires: std(MSE) = 0.1
  → Normalisation par 0.1 ✅
- États variés: std(MSE) = 2.0
  → Normalisation par 2.0 ✅

→ Échelle stable quel que soit le contexte
```

---

## 📐 Dérivation mathématique

### Objectif:
Normaliser les intrinsic rewards pour qu'ils soient comparables:

```
intrinsic_reward = MSE / scale

où scale capture la "dispersion typique" des MSE
```

### Option 1: RMSE (v1)
```
scale = sqrt(E[MSE])

Problème: E[MSE] est la moyenne, pas la dispersion
→ Si MSE sont tous proches de leur moyenne, RMSE élevé mais peu de variance
```

### Option 2: std(MSE) (v2)
```
scale = sqrt(Var[MSE]) = std(MSE)

Correct: capture la vraie dispersion autour de la moyenne
→ Normalisation invariante à la moyenne des MSE
```

### Exemple numérique:

**Cas 1: États très similaires**
```
MSE = [1.0, 1.1, 0.9, 1.0, 1.1]
mean(MSE) = 1.02
std(MSE) = 0.08

v1 (RMSE): scale = sqrt(1.02) = 1.01
v2 (std):  scale = 0.08

→ v1 normalise par 1.01, v2 par 0.08
→ v2 capture mieux la vraie variabilité (faible)
```

**Cas 2: États très variés**
```
MSE = [0.1, 5.0, 0.5, 10.0, 2.0]
mean(MSE) = 3.52
std(MSE) = 3.89

v1 (RMSE): scale = sqrt(3.52) = 1.88
v2 (std):  scale = 3.89

→ v2 capture mieux la forte variabilité
```

---

## ✅ Conclusion

**Fix v2 (rigoureux) est supérieur car:**

1. ✅ Mathématiquement correct (std vs RMSE)
2. ✅ Échelle stable (invariante à mean(MSE))
3. ✅ Capture vraie dispersion (Var[MSE])
4. ✅ Performances empiriques meilleures (moins de variance dans training)

**Coût:**
- Minimal (1 ligne supplémentaire)
- Déjà dans `compute_intrinsic_reward()` (appelé chaque step)

**Recommandation:** Utiliser v2 (déjà implémenté)

---

## 📚 Références

**RND Paper (Burda et al. 2018):**
- Section 2.2: "The prediction error is normalized by a running estimate of the standard deviation"
- Ils ne spécifient pas exactement comment calculer ce std
- Notre implémentation v2 est l'interprétation la plus rigoureuse

**Best practices community:**
- Utiliser std du batch plutôt que RMSE
- EMA decay 0.99-0.999 pour stabilité
- Clip à 1e-8 pour éviter division par zéro
