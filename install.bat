@echo off
chcp 65001 >nul
setlocal
cd /d %~dp0

echo ============================================================
echo   Meeting Bot - One-click Installer
echo ============================================================
echo.

rem ---------- 1. Locate Python 3.12 ----------
set "PY="
py -3.12 -V >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    python -V 2>nul | findstr /r "3\.1[2-9]" >nul && set "PY=python"
)
if not defined PY goto NO_PYTHON
echo [1/4] Python detected: %PY%

rem ---------- 2. Create venv ----------
if exist ".venv\Scripts\python.exe" goto HAVE_VENV
echo [2/4] Creating virtual environment ...
%PY% -m venv .venv
if errorlevel 1 goto VENV_FAIL
goto VENV_OK
:HAVE_VENV
echo [2/4] Virtual environment exists, skipping
:VENV_OK

rem ---------- 3. Install dependencies ----------
echo [3/4] Installing dependencies - about 1GB, first run takes minutes ...
.venv\Scripts\python -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\pip install -r requirements.txt
if not errorlevel 1 goto DEPS_OK
echo        Default index failed, retrying with Tsinghua mirror ...
.venv\Scripts\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto DEPS_FAIL
:DEPS_OK

rem ---------- 4. Configure + shortcut ----------
if exist ".env" goto HAVE_ENV
echo [4/4] First-time setup - please fill in your API keys ...
.venv\Scripts\python setup_env.py
if errorlevel 1 goto CFG_FAIL
goto ENV_OK
:HAVE_ENV
echo [4/4] .env exists, skipping setup
:ENV_OK

.venv\Scripts\python install_shortcut.py

echo.
echo ============================================================
echo   Done. A shortcut named "Meeting Translator" is on Desktop.
echo   Double-click it and click Start to test.
echo ============================================================
pause
exit /b 0

:NO_PYTHON
echo [ERROR] Python 3.12+ not found.
echo         Install from https://www.python.org/downloads/
echo         Remember to check "Add python.exe to PATH".
pause
exit /b 1

:VENV_FAIL
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:DEPS_FAIL
echo [ERROR] Dependency install failed. Check your network and retry.
pause
exit /b 1

:CFG_FAIL
echo [ERROR] Setup not finished. Re-run this script later.
pause
exit /b 1
