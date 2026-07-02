@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File source\run_train_qwen_supermix_v26_cognitive_activation.ps1 -BaseModel Qwen/Qwen2.5-0.5B-Instruct -OutputDir artifacts\qwen_supermix_enhanced_v26_cognitive_activation -Device cpu
