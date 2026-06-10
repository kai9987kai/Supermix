@echo off
setlocal
set PYTHONUNBUFFERED=1
cd /d "%~dp0source"
echo Training Titan-Dreamer v43 (warm-start from Supermix v27 champion)...
python finetune_chat.py ^
  --data "..\datasets\conversation_data.mega_reasoning_creative_v25_75582.jsonl" ^
  --weights "..\runtime_python\champion_model_chat_supermix_v27_500k_ft.pth" ^
  --output "champion_model_chat_titan_dreamer_v43_ft.pth" ^
  --meta "chat_model_meta_titan_dreamer_v43.json" ^
  --model_size titan_dreamer_expert ^
  --feature_mode context_v2 ^
  --epochs 4 ^
  --batch_size 32 ^
  --device auto > ..\train_titan_dreamer_v43_log.txt 2>&1
type ..\train_titan_dreamer_v43_log.txt
pause
endlocal
