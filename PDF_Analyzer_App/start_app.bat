@echo off
setlocal

REM Change to the directory of the script
cd /d "%~dp0"

REM Activate the Conda environment and run Streamlit
call conda activate torch260
streamlit run app.py

endlocal
