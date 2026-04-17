@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo Running model...
where py >nul 2>nul
if not errorlevel 1 (
  py -u run.py
  set "code=!ERRORLEVEL!"
) else (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -u run.py
    set "code=!ERRORLEVEL!"
  ) else (
    echo Could not find Python in PATH. Install Python or run from a shell where Python is available.
    set "code=9009"
  )
)
echo.
if not "!code!"=="0" (
  echo Script failed with exit code !code!.
) else (
  echo Script finished successfully.
)
echo.
pause
exit /b !code!
