@echo off
rem StrataRPCG: GLM-5.3-Flash over two PCs. Run this once on EACH PC (the 4090 PC and the 5070 Ti PC).
rem It runs Project Maya's setup (START-MAYA.bat) for pfeifferj's GSQ-RCO 3.5-bit GGUF (137.1 GB) and Maya's
rem vision files (1.1 GB): it checks the PC, compiles the engine (Visual Studio 2022 Build Tools and the CUDA
rem toolkit), asks before it downloads, builds the model's pack and writes its config - and does not start it.
rem Then start the split with scripts\stratarpcg\start-worker.ps1 (5070 Ti) and start-coordinator.ps1 (4090):
rem docs\STRATARPCG.md says how.
rem
rem Defaults: the model in ..\Strata-data\models-glm (next to this folder), a 512K context, no drafter download (the
rem split does not draft, and this file has no MTP block). Anything you add is passed on to the setup and wins, e.g.:
rem   START-HERE.bat --context 262144
rem   START-HERE.bat --models-dir D:\models
rem   START-HERE.bat --yes --download-model      (no questions, the download included)
rem   START-HERE.bat --model GSQ-RCO-3.0bit      (the smaller, faster 3.0-bit file, 117.5 GB, instead)
setlocal
cd /d "%~dp0"
set "MODELS=%~dp0..\Strata-data\models-glm"
set "CTX=--context 524288"
set "DIR=--models-dir "%MODELS%""
set "DRAFT=--draft mtp"
echo %* | findstr /i /c:"--context" >nul && set "CTX="
echo %* | findstr /i /c:"--models-dir" >nul && set "DIR="
echo %* | findstr /i /c:"--draft" >nul && set "DRAFT="
echo.
echo  StrataRPCG setup: GLM-5.3-Flash GSQ-RCO 3.5-bit, with pictures. Run it on both PCs.
echo.
call START-MAYA.bat --setup --model GSQ-RCO-3.5bit --no-start %CTX% %DIR% %DRAFT% %*
exit /b %errorlevel%
