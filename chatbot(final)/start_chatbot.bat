@echo off
cd /d "%~dp0"
echo Starting Chatbot Server... Please wait a moment.
echo A new browser window will open automatically when ready.
py -m streamlit run app.py
pause
