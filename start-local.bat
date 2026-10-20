@echo off
echo ========================================================
echo         Starting StudySphere AI on Localhost
echo ========================================================
echo.

:: Check for backend venv
if not exist "backend\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at backend\venv.
    echo Please ensure the backend virtual environment is created.
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "StudySphere AI - Backend Server" cmd /k "cd backend && venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Vite Frontend on http://localhost:8443 ...
start "StudySphere AI - Frontend Dev Server" cmd /k "npm run dev"

echo.
echo ========================================================
echo Both servers are launching!
echo - Frontend UI:   http://localhost:8443
echo - Backend API:   http://127.0.0.1:8000
echo - Swagger Docs:  http://127.0.0.1:8000/docs
echo ========================================================
echo.
pause
