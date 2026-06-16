@echo off
setlocal
cd /d "%~dp0"

set "OPT_CORES=10"
set "NEW20_BACKEND=avl"
set "NEW20_OBJECTIVE=q_g_per_ton_km"
set "NEW20_MISSION_L_KM=3000"
set "NEW20_MAX_CY=0.6"
set "NEW20_FIXED_H=500"
set "NEW20_FIXEDPOINT=0"
set "NEW20_PENALIZE_MZ=1"
set "NEW20_BIAS_MZ=0.001"
set "NEW20_ENABLE_M0_FEEDBACK=1"
set "NEW20_M0_SYNC_RELAX=0.5"
set "SHADE_MAX_GENERATIONS=0"
set "BRANCH_INIT_DIR=init_h500"
set "BRANCH_N_PER_BRANCH=140"
set "AVL_OUTDIR=gen_avl_logic1_qt_internaltrim"
set "ONLY_BRANCHES="
set "SHADE_INITIAL_POP="
set "SHADE_RESUME_STATE="

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 if exist "C:\ProgramData\Miniconda3\envs\vsppytools\python.exe" set "PYTHON_EXE=C:\ProgramData\Miniconda3\envs\vsppytools\python.exe"

if not exist "%BRANCH_INIT_DIR%\manifest.csv" (
  %PYTHON_EXE% src\generate_branch_initial_populations_v2.py --n-per-branch %BRANCH_N_PER_BRANCH% --outdir "%BRANCH_INIT_DIR%"
  if errorlevel 1 exit /b 1
)

echo [LOGIC1-QT-INTERNALTRIM-AVL] Start
echo PROJECT_ROOT=%cd%
echo OPT_CORES=%OPT_CORES%
echo NEW20_OBJECTIVE=%NEW20_OBJECTIVE%
echo NEW20_FIXED_H=%NEW20_FIXED_H%
echo AVL_OUTDIR=%AVL_OUTDIR%

%PYTHON_EXE% src\run_12branches_new20.py ^
  --backend avl ^
  --outdir "%AVL_OUTDIR%" ^
  --cores %OPT_CORES% ^
  --init-dir "%BRANCH_INIT_DIR%" ^
  --n-per-branch %BRANCH_N_PER_BRANCH%
exit /b %errorlevel%
