@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" || goto failed
if not exist app.py (
    echo Extract all files from the project ZIP before starting the app.
    goto failed
)

set "PYTHON_EXE="
set "PYTHON_FLAGS="
if exist "%USERPROFILE%\anaconda3\python.exe" call :try_python "%USERPROFILE%\anaconda3\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\python.exe" call :try_python "%USERPROFILE%\miniconda3\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where.exe python.exe py.exe 2^>nul') do (
        if not defined PYTHON_EXE call :try_python "%%P"
    )
)
if not defined PYTHON_EXE (
    echo No working Python 3.9 or newer was found. Install Python or Anaconda, then try again.
    goto failed
)

if /i "%~1"=="--demo" goto demo
if exist "data\imdb_local.jsonl" goto run
set "REVIEW_ARCHIVE="
if exist "data\aclImdb_v1.tar.gz" set "REVIEW_ARCHIVE=data\aclImdb_v1.tar.gz"
if defined REVIEW_ARCHIVE goto import
for %%I in ("%USERPROFILE%\tensorflow_datasets\downloads\imdb_reviews\*aclImdb_v1*.tar.gz") do (
    if exist "%%~fI" set "REVIEW_ARCHIVE=%%~fI"
)
if defined REVIEW_ARCHIVE goto import
echo Add aclImdb_v1.tar.gz to the data folder, or start with Start-Demo.cmd.
goto failed

:import
echo Importing local IMDb reviews. Please wait...
"%PYTHON_EXE%" %PYTHON_FLAGS% -B import_imdb.py "%REVIEW_ARCHIVE%" data/imdb_local.jsonl --split test --max-movies 12 --titles data/imdb_titles.json
if errorlevel 1 goto failed

:run
set "SEMANTIC_FLAG="
if /i "%~1"=="--semantic" (
    "%PYTHON_EXE%" %PYTHON_FLAGS% -B prepare_semantic.py
    if errorlevel 1 goto failed
    set "SEMANTIC_FLAG=--semantic"
)
"%PYTHON_EXE%" %PYTHON_FLAGS% -B app.py --corpus data/imdb_local.jsonl --open-browser %SEMANTIC_FLAG%
if errorlevel 1 goto failed
exit /b 0

:demo
"%PYTHON_EXE%" %PYTHON_FLAGS% -B app.py --open-browser
if errorlevel 1 goto failed
exit /b 0

:failed
echo The app could not start. See the message above.
pause
exit /b 1

:try_python
rem Windows Store installer aliases are zero-byte entries, not Python installations.
if "%~z1"=="0" exit /b 1
set "PROBE_FLAGS="
if /i "%~nx1"=="py.exe" set "PROBE_FLAGS=-3"
"%~1" %PROBE_FLAGS% -B -c "import sys; raise SystemExit(sys.version_info < (3, 9))" >nul 2>nul
if errorlevel 1 exit /b 1
set "PYTHON_EXE=%~f1"
set "PYTHON_FLAGS=%PROBE_FLAGS%"
exit /b 0
