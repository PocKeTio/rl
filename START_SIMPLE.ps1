# START SIMPLE TRAINING
# Config basée sur Google Research Football paper

Write-Host "=== SIMPLE TRAINING (Google Research Football paper) ===" -ForegroundColor Green
Write-Host "Config: train_simple.yaml" -ForegroundColor Cyan
Write-Host "Rewards: Checkpoint rewards (+0.1 for ball proximity)" -ForegroundColor Cyan
Write-Host "RND: Disabled" -ForegroundColor Cyan
Write-Host ""

.\.venv\Scripts\python.exe -m gfrl.cli.train_ppo_rnd --config configs/train_simple.yaml
