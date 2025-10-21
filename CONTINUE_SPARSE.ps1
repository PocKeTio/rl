# CONTINUE SPARSE - Reprend l'entraînement depuis le dernier checkpoint

Write-Host "=" -ForegroundColor Cyan
Write-Host "CONTINUE SPARSE TRAINING" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan

# Vérifier que le checkpoint existe
if (Test-Path "checkpoints\last.pt") {
    Write-Host "`nCheckpoint trouvé:" -ForegroundColor Green
    Write-Host "  - File: checkpoints\last.pt" -ForegroundColor White
    
    # Afficher les infos du checkpoint (optionnel)
    Write-Host "`nReprise de l'entraînement..." -ForegroundColor Yellow
    Write-Host "  - Config: train_sparse.yaml" -ForegroundColor White
    Write-Host "  - Rewards: rewards_sparse.yaml (sparse + RND)" -ForegroundColor White
    Write-Host ""
    
    # Lancer avec --resume
    .\.venv\Scripts\python.exe -m gfrl.cli.train_ppo_rnd --config configs\train_sparse.yaml --resume checkpoints\last.pt
    
} else {
    Write-Host "`n[ERREUR] Aucun checkpoint trouvé!" -ForegroundColor Red
    Write-Host "  Vérifie que checkpoints\last.pt existe" -ForegroundColor Yellow
    Write-Host "  Utilise START_SPARSE.ps1 pour commencer un nouveau training" -ForegroundColor Yellow
}
