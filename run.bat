@echo off
REM EgoCut - Windows entry point, same contract as run.sh:   run.bat <video.mp4> <out.json>
REM Needs Python 3.10-3.12 with requirements.txt installed and ffmpeg on PATH (or imageio-ffmpeg).
REM Weights: run "bash weights/download_weights.sh" once (Git Bash or WSL), or download the files listed
REM in weights\download_weights.sh from the GitHub release weights-v1 into the weights\ folder.
setlocal
if "%~2"=="" (echo usage: run.bat ^<video.mp4^> ^<out.json^> & exit /b 2)
set HERE=%~dp0
if "%PYTHON%"=="" set PYTHON=python
if not exist "%~dp2" mkdir "%~dp2"
"%PYTHON%" "%HERE%code\egocut_infer.py" "%~1" "%~2"
if errorlevel 1 echo [run.bat] main pipeline exited with an error 1>&2
"%PYTHON%" "%HERE%code\fallback.py" "%~1" "%~2"
endlocal
