@echo off
REM AutoEncoder 모델 평가 실행 스크립트

echo ========================================
echo AutoEncoder Model Evaluation
echo ========================================
echo.

REM 프로젝트 루트로 이동 (배치 파일 위치에서 3단계 위)
cd /d "%~dp0..\..\..\"

echo Current directory: %CD%
echo.

REM 기본 경로 설정
set MODEL_PATH=C:\Users\user\Workspace\datasets@20260109\autoencoder\model.pt
set DB_PATH=C:\Users\user\Workspace\datasets@20260109\datasets_norm_all.duckdb
set OUTPUT_DIR=scripts\pre\diagnosis\results
set DEVICE=cuda
set MAX_SAMPLES=10000

echo Model: %MODEL_PATH%
echo Database: %DB_PATH%
echo Output: %OUTPUT_DIR%
echo Device: %DEVICE%
echo Max Samples: %MAX_SAMPLES%
echo.

REM Python 스크립트 실행
python scripts\pre\diagnosis\evaluate_autoencoder.py ^
    --model "%MODEL_PATH%" ^
    --db "%DB_PATH%" ^
    --output "%OUTPUT_DIR%" ^
    --device %DEVICE% ^
    --max-samples %MAX_SAMPLES%

echo.
echo ========================================
echo Evaluation Complete!
echo Results saved to: %OUTPUT_DIR%
echo ========================================
pause
