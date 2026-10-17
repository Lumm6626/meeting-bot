@echo off
chcp 65001 >nul
cd /d %~dp0
"%~dp0..\.venv\Scripts\python.exe" -X utf8 "%~dp0diagnose_mic.py"
pause
