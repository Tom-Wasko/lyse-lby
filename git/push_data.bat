@echo off
setlocal

REM ===== SETTINGS =====
set REPO_URL=git@github.com:sztunter2/tlustes.git
set BRANCH=main
set DATA_DIR=data
set TEMP_DIR=_push_repo

echo ================================
echo Preparing push to tlustes repo
echo ================================

REM Remove old temp repo if exists
if exist %TEMP_DIR% (
    echo Removing old temp repo...
    rmdir /s /q %TEMP_DIR%
)

REM Clone fresh copy
echo Cloning repository...
git clone %REPO_URL% %TEMP_DIR%
if errorlevel 1 (
    echo Clone failed.
    exit /b 1
)

cd %TEMP_DIR%

REM Clean tlustes/data directory
if exist %DATA_DIR% (
    echo Cleaning remote data directory...
    rmdir /s /q %DATA_DIR%
)

mkdir %DATA_DIR%

REM Copy new data from parent git\data
echo Copying new data...
xcopy "..\%DATA_DIR%\*" "%DATA_DIR%\" /E /I /Y

REM Commit and push
git add .
git commit -m "Update data folder"
git push origin %BRANCH%

echo ================================
echo Push completed.
echo ================================

cd ..
rmdir /s /q %TEMP_DIR%

endlocal
pause
