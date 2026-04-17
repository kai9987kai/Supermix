@echo off
setlocal
cd /d "%~dp0"
python qwen_chat_web_app.py --port 8010 --device cpu --adapter_dir ..\artifacts\qwen_supermix_enhanced_v8_repair\adapter
endlocal
