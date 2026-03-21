@echo off
setlocal
if defined SUPERMIX_PYTHON if exist "%SUPERMIX_PYTHON%" (
  "%SUPERMIX_PYTHON%" source\training_monitor_gui.py --root .
  goto :eof
)
if exist ".venv-dml\Scripts\python.exe" (
  ".venv-dml\Scripts\python.exe" source\training_monitor_gui.py --root .
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" source\training_monitor_gui.py --root .
) else (
  python source\training_monitor_gui.py --root .
)
