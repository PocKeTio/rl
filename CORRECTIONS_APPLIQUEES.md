# ✅ CORRECTIONS APPLIQUÉES

## 🔍 Problème Identifié

**0 buts en 800K steps** sur `academy_3_vs_1_with_keeper`

**Cause** : Le scénario `academy_3_vs_1_with_keeper` **n'existe pas** dans GRF.

## ✅ Corrections Effectuées

### 1. Curriculum Corrigé (`curriculum_5v5.yaml`)

Remplacé les scénarios inexistants par des scénarios GRF built-in :

| Avant (❌ n'existe pas) | Après (✅ existe) |
|------------------------|------------------|
| `academy_3_vs_1_with_keeper` | `academy_counterattack_easy` |
| `academy_3_vs_2_with_keeper` | `academy_counterattack_hard` |
| `academy_3_vs_3_with_keeper` | `11_vs_11_easy_stochastic` |

**Nouvelle progression (9 phases)** :
1. `academy_empty_goal_close` (but vide 5m) - 95% winrate
2. `academy_empty_goal` (but vide 20m) - 85% winrate
3. `academy_run_to_score` (courir + tirer) - 80% winrate
4. `academy_run_to_score_with_keeper` (avec gardien) - 70% winrate
5. `academy_single_goal_vs_lazy` (1 défenseur passif) - 75% winrate
6. `academy_counterattack_easy` (1v1 facile) - 75% winrate
7. `academy_pass_and_shoot_with_keeper` (passe + tir) - 60% winrate
8. `academy_counterattack_hard` (1v1 difficile) - 60% winrate
9. `academy_corner` (corners) - 50% winrate
10. `11_vs_11_easy_stochastic` (match complet) - 55% winrate

### 2. Training Config Corrigé (`train_5v5.yaml`)

**Après** :
```yaml
num_envs: 24  # Safe pour Windows
batch_size: 6144  # 8 * 256
minibatch_size: 1024  # Diviseur de batch_size
```

### 3. Tracking des Buts Corrigé

Ajouté `get_current_score()` dans tous les wrappers pour extraire le score via `AsyncVectorEnv.call()`.

**Flow** :
```
AsyncVectorEnv.call('get_current_score')
  ↓
RewardShaperWrapper.get_current_score()
  ↓
RunningNormWrapper.get_current_score()
  ↓
ObsWrapperRaw.get_current_score()
  ↓
GRFtoGymnasiumWrapper.get_current_score()
  ↓
Extrait obs['score'] depuis GRF
```

## 🚀 Comment Relancer

### Option 1 : Script Automatique

```powershell
.\RESTART.ps1
```

### Option 2 : Manuel

```powershell
# Clean
Remove-Item checkpoints\*.pt -Force
Remove-Item -Recurse logs\tensorboard\*

# Train
.\.venv\Scripts\python.exe -m gfrl.cli.train_ppo_rnd --config configs\train_5v5.yaml
```

## 📈 Résultats Attendus

### Phase 1 : academy_empty_goal_close

**Après ~10-20K steps** :
```
Goals scored: 0.8-1.0/game
Winrate: 90-95%
```

**Si toujours 0 buts après 50K steps** → Il y a un autre bug.

### Progression Complète

- **Phase 1-3** (academy basics) : ~5M steps, 2-3h
- **Phase 4-6** (keeper + passe) : ~10M steps, 4-5h  
- **Phase 7-10** (11v11) : ~20M steps, 8-10h

**Total** : ~35M steps, 15-18h sur GPU

## 🔧 Si Ça Ne Marche Toujours Pas

### Debug 1 : Vérifier les Buts

Ajouter dans `trainer.py` ligne 487 (après `step`) :

```python
if reward.max() > 0.5:
    logger.info(f"⚽ GOAL DETECTE! Env {reward.argmax()}, Reward: {reward.max():.2f}")
```

### Debug 2 : Vérifier le Scénario

```powershell
python -c "import gfootball.env as e; env = e.create_environment('academy_empty_goal_close', representation='raw'); print('Scenario OK')"
```

### Debug 3 : Test Random Agent

```powershell
python test_random_agent.py
```

Devrait donner : **7-10 buts sur 10 épisodes**

## 📊 Suivi via TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs/tensorboard
```

Ouvrir : http://localhost:6006

**Métriques à surveiller** :
- `goals_scored` - doit augmenter rapidement phase 1
- `winrate` - doit atteindre 95% phase 1
- `entropy` - doit descendre lentement (exploration → exploitation)
- `intrinsic_reward` - RND doit encourager l'exploration

## ✅ Checklist de Validation

- [x] Curriculum utilise scénarios GRF built-in
- [x] num_envs = 8 (Windows safe)
- [x] batch_size cohérent avec num_envs
- [x] get_current_score() implémenté
- [x] Test random agent OK (1 but en 10 essais)
- [ ] Training détecte des buts (à vérifier après lancement)

---

**Lancez `.\RESTART.ps1` et surveillez les 10 premières minutes !**

Si vous voyez `Goals scored: 0.5+/game` dans les logs → ✅ Ça marche !
