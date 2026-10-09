@echo off
chcp 65001 >nul
cd /d "%~dp0"

set JEFF_CHECKPOINT=checkpoints\jeff-0.8b
set PORT=8765

echo ============================================
echo  Jeff decision model server
echo  checkpoint: %JEFF_CHECKPOINT%
echo  endpoint  : http://127.0.0.1:%PORT%/v1/systemone
echo  press Ctrl+C to stop
echo ============================================

".venv\Scripts\jeff-serve.exe"

pause
