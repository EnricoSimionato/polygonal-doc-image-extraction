@echo off
rem Windows: double-click to start the app. First run creates .venv and installs the requirements.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Setting up the Python environment in .venv (first run only)...
  py -3 -m venv .venv || python -m venv .venv || (echo Python 3.10 or newer is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q -r requirements.txt || (echo Installation failed & pause & exit /b 1)
)
.venv\Scripts\python app\server.py
pause
