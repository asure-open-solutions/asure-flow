@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo   AsureFlow - Server
echo   ===================
echo.

rem Prefer the project virtual environment. Fall back to system Python/py only for first setup.
set "PY_EXE=%~dp0server\.venv\Scripts\python.exe"
if exist "%PY_EXE%" goto have_python

where python >nul 2>nul
if not errorlevel 1 (
    set "PY_BOOT=python"
    goto setup_server
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PY_BOOT=py -3"
    goto setup_server
)

echo   ERROR: Python is not installed or not on PATH.
echo   Install Python 3.11+ from https://www.python.org/downloads/
echo.
pause
exit /b 1

:setup_server
echo   [setup] Creating server virtual environment...
cd /d "%~dp0server"
%PY_BOOT% -m venv .venv
if errorlevel 1 (
    echo   ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 (
    echo   ERROR: Failed to install server dependencies.
    pause
    exit /b 1
)
cd /d "%~dp0"
set "PY_EXE=%~dp0server\.venv\Scripts\python.exe"

:have_python
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" set "%%a=%%b"
    )
)
if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8000"

echo   Python: %PY_EXE%
echo   Local:  http://localhost:%PORT%
echo   Docs:   http://localhost:%PORT%/docs
echo.
echo   Press Ctrl+C to stop.
echo.

set "PYTHONPATH=%~dp0server\src"
cd /d "%~dp0server"
"%PY_EXE%" -m uvicorn asure_flow.main:app --host %HOST% --port %PORT% --reload --reload-exclude .venv --ws-max-size 1048576
pause
exit /b %errorlevel%
