@echo off
echo ================================
echo   GitHub Auto Push Script
echo ================================
echo.

REM Initialize git if needed
git init

REM Set main branch
git branch -M main

REM Add remote (ignore error if it already exists)
git remote add origin [git@github.com](mailto:git@github.com):Tom-Wasko/lyse-lby.git 2>nul

REM Add all files
git pull origin main --rebase
git add .
git commit -m "Auto commit %date% %time%" 2>nul
git push origin main

echo.
echo ================================
echo   Push completed
echo ================================
pause
