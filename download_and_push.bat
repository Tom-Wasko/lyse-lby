@echo off

echo Downloading stock data..
python "C:\Users\stunt\lyse-lby\update.py"

echo Finding signals
python find_data.py

echo Push to git...
call git\push_data.bat

echo All tasks completed.
pause