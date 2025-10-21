# TRAINING SPARSE - Laisse l'agent apprendre SEUL
# Seulement rewards: goals + wins/losses
# L'agent explore naturellement grâce à entropy élevée

Write-Host "=" -ForegroundColor Cyan
Write-Host "TRAINING SPARSE - Apprentissage AUTONOME" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan

Write-Host "`nPhilosophie:" -ForegroundColor Yellow
Write-Host "  - Rewards: SEULEMENT goals (+100) et wins (+10)" -ForegroundColor White
Write-Host "  - RND (curiosity): Active pour exploration autonome" -ForegroundColor White
Write-Host "  - Entropy: ÉLEVÉE (0.1) pour exploration maximale" -ForegroundColor White
Write-Host "  - Durée: 100M steps (peut tourner 2-3 jours)" -ForegroundColor White
Write-Host "  - L'agent découvre SEUL comment marquer" -ForegroundColor White

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  - Config: train_sparse.yaml" -ForegroundColor White
Write-Host "  - Num envs: 24" -ForegroundColor White
Write-Host "  - Rewards: rewards_sparse.yaml (goals only)" -ForegroundColor White
Write-Host "  - Curriculum: academy_3_vs_1_with_keeper -> 5v5" -ForegroundColor White

Write-Host "`n⚠️  IMPORTANT:" -ForegroundColor Red
Write-Host "  - L'apprentissage sera LENT au début (normal)" -ForegroundColor White
Write-Host "  - Premiers goals: après 5-10M steps" -ForegroundColor White
Write-Host "  - Winrate significatif: après 20-30M steps" -ForegroundColor White
Write-Host "  - PATIENCE = CLÉ!" -ForegroundColor White

Write-Host "`nNettoyage des checkpoints..." -ForegroundColor Yellow
Remove-Item -Force checkpoints\*.pt -ErrorAction SilentlyContinue
Write-Host "OK" -ForegroundColor Green

Write-Host "`nLancement training..." -ForegroundColor Yellow
Write-Host ""

.\.venv\Scripts\python.exe -m gfrl.cli.train_ppo_rnd --config configs\train_sparse.yaml
