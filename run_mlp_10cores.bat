@echo off
setlocal
cd /d "%~dp0"

set "OPT_CORES=10"
set "NEW20_BACKEND=mlp"
set "NEW20_OBJECTIVE=q_g_per_ton_km"
set "NEW20_MISSION_L_KM=3000"
set "NEW20_MAX_CY=0.6"
set "NEW20_FIXEDPOINT=0"
set "NEW20_PENALIZE_MZ=1"
set "NEW20_BIAS_MZ=0.0015"
set "NEW20_ENABLE_M0_FEEDBACK=1"
set "NEW20_M0_SYNC_RELAX=0.5"
set "NEW20_ENABLE_SPREAD_STOP=1"
set "NEW20_SPREAD_STOP_MIN_FEASIBLE_RATIO=0.1"
set "SHADE_MAX_GENERATIONS=0"
set "BRANCH_INIT_DIR=init_h500"
set "BRANCH_N_PER_BRANCH=140"
set "OUTDIR=gen_mlp_original12_oldbranch_qt_np140_dampmz20"
set "NEW20_FIXED_H=500"
set "NEW20_MLP_BATCH=1"
set "AERO_MODEL_DIR_NORMAL=models\aero_mlp_original12_normal_oldbranch_300k"
set "AERO_MODEL_DIR_DUCK=models\aero_mlp_original12_duck_oldbranch_300k"
set "NEW20_MODEL_DIR_NORMAL=%AERO_MODEL_DIR_NORMAL%"
set "NEW20_MODEL_DIR_DUCK=%AERO_MODEL_DIR_DUCK%"
set "NEW20_PENALIZE_MZ_OMEGAZ=1"
set "NEW20_MIN_MZ_OMEGAZ=-20"
if not defined ONLY_BRANCHES set "ONLY_BRANCHES="
if not defined SHADE_INITIAL_POP set "SHADE_INITIAL_POP="
if not defined SHADE_RESUME_STATE set "SHADE_RESUME_STATE="
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>nul
if errorlevel 1 if exist "C:\ProgramData\Miniconda3\envs\vsppytools\python.exe" set "PYTHON_EXE=C:\ProgramData\Miniconda3\envs\vsppytools\python.exe"

if not exist "%BRANCH_INIT_DIR%\manifest.csv" (
  %PYTHON_EXE% src\generate_branch_initial_populations_v2.py --n-per-branch %BRANCH_N_PER_BRANCH% --outdir "%BRANCH_INIT_DIR%"
  if errorlevel 1 exit /b 1
)
if not exist "%AERO_MODEL_DIR_NORMAL%\aero_mlp_v2_best.keras" (
  echo Missing %AERO_MODEL_DIR_NORMAL%\aero_mlp_v2_best.keras
  exit /b 1
)
if not exist "%AERO_MODEL_DIR_DUCK%\aero_mlp_v2_best.keras" (
  echo Missing %AERO_MODEL_DIR_DUCK%\aero_mlp_v2_best.keras
  exit /b 1
)
echo [MLP-DAMPMZ20-QT] Start
%PYTHON_EXE% src\run_12branches_new20.py ^
  --backend mlp ^
  --outdir "%OUTDIR%" ^
  --model-dir "%AERO_MODEL_DIR_NORMAL%" ^
  --cores %OPT_CORES% ^
  --init-dir "%BRANCH_INIT_DIR%" ^
  --n-per-branch %BRANCH_N_PER_BRANCH%
exit /b %errorlevel%


