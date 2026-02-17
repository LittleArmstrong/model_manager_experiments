@echo off
REM Zum Projektordner wechseln (Ordner der .bat Datei)
cd /d %~dp0

REM Virtuelle Umgebung aktivieren
call .venv\Scripts\activate.bat

REM Python Script ausführen
python main.py

@REM REM Fenster offen halten (optional)
@REM pause
