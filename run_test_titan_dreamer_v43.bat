@echo off
setlocal
cd /d "%~dp0"
echo Running Titan-Dreamer v43 smoke test...
python source\test_titan_dreamer_expert.py > test_titan_dreamer_output.txt 2>&1
type test_titan_dreamer_output.txt
echo.
echo Output saved to test_titan_dreamer_output.txt
pause
endlocal
