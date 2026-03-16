@echo off
setlocal enabledelayedexpansion

set "LOG=%~dp0push_data.log"
call :main > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
echo.
echo Exit code: %RC%
pause
exit /b %RC%

:main
REM ================================
REM SETTINGS
REM ================================

set "REPO_URL=https://github.com/sztunter2/tlustes.git"
set "BRANCH=main"
set "TEMP_DIR=%~dp0_push_repo"
set "KEEP_TEMP=1"

set "FOLDER_PATTERN=data_*"
set "SIGNALS_FILE=signals.csv"
set "ROBO_XF=CON.* PRN.* AUX.* NUL.* COM1.* COM2.* COM3.* COM4.* COM5.* COM6.* COM7.* COM8.* COM9.* LPT1.* LPT2.* LPT3.* LPT4.* LPT5.* LPT6.* LPT7.* LPT8.* LPT9.*"

echo ======================================
echo Preparing clean push to tlustes repository
echo ======================================

REM ================================
REM PREPARE TEMP REPO
REM ================================
if exist "%TEMP_DIR%\.git" (
    echo Using existing temporary repo...
) else (
    if exist "%TEMP_DIR%" (
        echo Removing invalid temp folder...
        rmdir /s /q "%TEMP_DIR%"
    )
    echo Cloning repository...
    git clone "%REPO_URL%" "%TEMP_DIR%"
    if errorlevel 1 (
        echo ERROR: Clone failed.
        exit /b 1
    )
)

cd /d "%TEMP_DIR%"
set "GIT_SAFE=-c safe.directory=%TEMP_DIR%"
set "GIT=git %GIT_SAFE%"
%GIT% config core.longpaths true
%GIT% remote set-url origin "%REPO_URL%" >nul 2>nul

echo Fetching latest from origin...
%GIT% fetch origin
if errorlevel 1 (
    echo WARNING: Fetch failed. Continuing with existing repo state.
)

%GIT% checkout "%BRANCH%" >nul 2>nul
if errorlevel 1 (
    %GIT% checkout -b "%BRANCH%"
    if errorlevel 1 (
        echo ERROR: Could not checkout branch "%BRANCH%".
        exit /b 1
    )
)

REM ================================
REM CLEAN WHOLE REPO (KEEP .git)
REM ================================
echo Cleaning repository working tree...
for /f "delims=" %%D in ('dir /b /a:d 2^>nul') do (
    if /i not "%%D"==".git" rmdir /s /q "%%D"
)
for /f "delims=" %%F in ('dir /b /a:-d 2^>nul') do (
    del /f /q "%%F"
)

REM ================================
REM COPY NEW DATA
REM ================================
echo Copying new data folders...
set "ROOT=%~dp0"
set "MISSING=0"
set "FOUND=0"
echo NOTE: Skipping Windows-reserved filenames (CON, PRN, AUX, NUL, COM1-9, LPT1-9).

for /d %%F in ("%ROOT%%FOLDER_PATTERN%") do (
    set "FOUND=1"
    set "NAME=%%~nxF"
    echo Copying !NAME!...
    robocopy "%%F" "!NAME!" /E /XF %ROBO_XF%
    set "RC=!ERRORLEVEL!"
    if !RC! GEQ 8 (
        echo ERROR: Robocopy failed for !NAME! with code !RC!.
        exit /b 1
    )
)

if "%FOUND%"=="0" (
    echo ERROR: No source folders matching "%ROOT%%FOLDER_PATTERN%".
    exit /b 1
)

if exist "%ROOT%%SIGNALS_FILE%" (
    echo Copying %SIGNALS_FILE%...
    copy /y "%ROOT%%SIGNALS_FILE%" "%SIGNALS_FILE%" >nul
) else (
    echo ERROR: Source file %SIGNALS_FILE% does not exist at "%ROOT%%SIGNALS_FILE%".
    set "MISSING=1"
)

if "%MISSING%"=="1" exit /b 1


REM ================================
REM COMMIT AND PUSH
REM ================================
echo Adding files to git...
%GIT% add -A

set "NEED_PUSH=1"

%GIT% rev-parse --verify HEAD >nul 2>nul
if errorlevel 1 (
    %GIT% commit -m "Replace data folders and signals.csv"
    if errorlevel 1 (
        echo ERROR: Commit failed.
        exit /b 1
    )
) else (
    %GIT% diff-index --quiet HEAD
    if errorlevel 1 (
        %GIT% commit -m "Replace data folders and signals.csv"
        if errorlevel 1 (
            echo ERROR: Commit failed.
            exit /b 1
        )
    ) else (
        echo No changes to commit.
    )
)

echo Pushing to GitHub...
%GIT% push origin "%BRANCH%"
if errorlevel 1 (
    echo ERROR: Push failed.
    exit /b 1
)
echo ======================================
echo Push completed successfully
echo ======================================

if "%KEEP_TEMP%"=="0" (
    cd /d "%~dp0"
    rmdir /s /q "%TEMP_DIR%"
)

endlocal
exit /b 0
