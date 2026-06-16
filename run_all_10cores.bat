@echo off
setlocal
cd /d "%~dp0"
echo [LOGIC1-QT-INTERNALTRIM] MLP -> AVL
call "%~dp0run_mlp_10cores.bat"
if errorlevel 1 goto :err
call "%~dp0run_avl_10cores.bat"
if errorlevel 1 goto :err
echo [LOGIC1-QT-INTERNALTRIM] Done
exit /b 0
:err
echo [LOGIC1-QT-INTERNALTRIM] Failed
exit /b 1
