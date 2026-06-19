@echo off
setlocal
cd /d "%~dp0"

set "OPT_CORES=10"
set "NEW20_BACKEND=mlp"
set "NEW20_OBJECTIVE=q_g_per_ton_km"
set "NEW20_MISSION_L_KM=3000"
set "NEW20_MAX_CY=0.6"
set "NEW20_FIXED_H=500"
set "NEW20_FIXEDPOINT=0"
set "NEW20_PENALIZE_MZ=1"
set "NEW20_BIAS_MZ=0.0015"
set "NEW20_ENABLE_M0_FEEDBACK=1"
set "NEW20_M0_SYNC_RELAX=0.5"
set "NEW20_USE_DIRECT_K=1"
set "NEW20_MLP_BATCH=1"
set "SHADE_MAX_GENERATIONS=0"
set "BRANCH_INIT_DIR=init_h500"
set "BRANCH_N_PER_BRANCH=140"
set "MLP_OUTDIR=gen_mlp_logic1_qt_internaltrim"
set "AERO_MODEL_DIR_NORMAL=models\aero_mlp_original12_normal_qkhead_300k_hard40k"
set "AERO_MODEL_DIR_DUCK=models\aero_mlp_original12_duck_qkhead_300k_hard40k"
set "NEW20_MODEL_DIR_NORMAL=%AERO_MODEL_DIR_NORMAL%"
set "NEW20_MODEL_DIR_DUCK=%AERO_MODEL_DIR_DUCK%"
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

echo [LOGIC1-QT-INTERNALTRIM-MLP] Start
echo PROJECT_ROOT=%cd%
echo OPT_CORES=%OPT_CORES%
echo NEW20_OBJECTIVE=%NEW20_OBJECTIVE%
echo NEW20_MLP_BATCH=%NEW20_MLP_BATCH%
echo NEW20_FIXED_H=%NEW20_FIXED_H%
echo MLP_OUTDIR=%MLP_OUTDIR%

%PYTHON_EXE% src\run_12branches_new20.py ^
  --backend mlp ^
  --outdir "%MLP_OUTDIR%" ^
  --model-dir "%AERO_MODEL_DIR_NORMAL%" ^
  --cores %OPT_CORES% ^
  --init-dir "%BRANCH_INIT_DIR%" ^
  --n-per-branch %BRANCH_N_PER_BRANCH%
exit /b %errorlevel%
