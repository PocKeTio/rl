# Script de redémarrage propre

Write-Host "=" -ForegroundColor Cyan
Write-Host "NETTOYAGE ET REDEMARRAGE" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan

# Clean checkpoints
Write-Host "`n1. Nettoyage checkpoints..." -ForegroundColor Yellow
Remove-Item -Force checkpoints\*.pt -ErrorAction SilentlyContinue
Write-Host "   OK" -ForegroundColor Green

# Clean logs
Write-Host "`n2. Nettoyage logs..." -ForegroundColor Yellow
Remove-Item -Recurse -Force logs\tensorboard\* -ErrorAction SilentlyContinue
Write-Host "   OK" -ForegroundColor Green

# Afficher la config
Write-Host "`n3. Configuration:" -ForegroundColor Yellow
Write-Host "   - Config: train_5v5.yaml" -ForegroundColor White
Write-Host "   - Num envs: 24 (défini dans train_5v5.yaml)" -ForegroundColor White
Write-Host "   - Curriculum: academy_3_vs_1_with_keeper -> 5v5" -ForegroundColor White
Write-Host "   - Rewards: dense (rewards_dense.yaml)" -ForegroundColor White

# Lancer training
Write-Host "`n4. Lancement training..." -ForegroundColor Yellow
Write-Host ""

.\.venv\Scripts\python.exe -m gfrl.cli.train_ppo_rnd --config configs\train_5v5.yaml
