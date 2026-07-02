@echo off
call run_mlp_10cores.bat
if errorlevel 1 exit /b 1
call run_avl_10cores.bat
exit /b %errorlevel%
