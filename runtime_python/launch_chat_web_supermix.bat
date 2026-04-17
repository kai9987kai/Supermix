@echo off
setlocal
cd /d "%~dp0"
python chat_web_app.py --weights champion_model_chat_supermix_v27_500k_ft.pth --meta chat_model_meta_supermix_v27_500k.json --autoload --port 8000
endlocal
