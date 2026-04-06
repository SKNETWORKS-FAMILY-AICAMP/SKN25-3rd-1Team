@echo off
echo Starting Agentic Mobile CS API and Streamlit UI...

:: Start FastAPI Backend (Takes ~30 seconds to load)
start "FastAPI Server" cmd /K "set PYTHONPATH=. && .venv\Scripts\python.exe -m uvicorn entrypoint.main:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] FastAPI Server will take ~30 seconds to start.
echo        Please wait until "Application startup complete" appears.

:: Start Streamlit UI
start "Streamlit UI" cmd /K "set PYTHONPATH=. && .venv\Scripts\python.exe -m streamlit run frontend/app.py"

echo Both servers are started in separate windows!
