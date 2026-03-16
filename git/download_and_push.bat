@echo off

cd /d "%~dp0"

echo ===============================
echo Starting pipeline
echo ===============================

echo Running update_all...
call "%~dp0..\all_data\update_all.bat"

echo Running find_data.py...
python "%~dp0find_data.py"

echo Running push_data...
call "%~dp0push_data.bat"

echo ===============================
echo Pipeline finished
echo ===============================

pause
