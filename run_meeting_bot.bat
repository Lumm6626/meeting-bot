@echo off
chcp 65001 >nul
cd /d %~dp0
"%~dp0.venv\Scripts\python.exe" -m meeting_bot
if errorlevel 1 pause
