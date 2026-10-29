@echo off
chcp 65001 >nul
title Transformer 2017 — 3D Visual Simulator (ARGON-TX7)
color 0B

echo =========================================================================
echo       TRANSFORMER 2017 — 3D INTERACTIVE VISUAL SIMULATOR
echo              Grounded Keras 3 + TensorFlow Backend
echo                    ARGON A7 Visual Assembly
echo =========================================================================
echo.

cd /d "%~dp0"

echo [1/3] Khởi động FastAPI Backend (Keras 3 Transformer 2017)...
start "ARGON Backend [Port 8000]" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && python backend/main.py"

timeout /t 3 /nobreak >nul

if not exist "%~dp0frontend\node_modules" (
    echo [2/3] Chưa tìm thấy node_modules, đang cài đặt thư viện frontend...
    cd /d "%~dp0frontend"
    call npm install
    cd /d "%~dp0"
)

echo [2/3] Khởi động Vite 3D Web Frontend (Three.js / React Three Fiber)...
start "ARGON Frontend [Port 5173]" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 3 /nobreak >nul

echo [3/3] Đang mở trình duyệt tại: http://localhost:5173/ ...
start http://localhost:5173/

echo.
echo =========================================================================
echo   SIMULATOR ĐANG CHẠY THÀNH CÔNG!
echo   - 3D Web UI: http://localhost:5173/
echo   - Backend API: http://localhost:8000/api/health
echo   - Đóng các cửa sổ terminal để dừng ứng dụng.
echo =========================================================================
pause
