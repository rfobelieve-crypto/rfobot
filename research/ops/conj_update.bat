@echo off
REM Daily updater for the conjunction-event forward clock.
REM Registration: research/poc/conj_clock.py  (frozen 2026-09-07)
REM Chain: fetch 1m klines -> rebuild bars parquet -> fetch OI -> score clock.
REM It self-reports {ok,reason} to research/poc/data/results/conj_clock_last.json
REM which the freshness board tracks as "conj clock flag".
REM
REM ASCII ONLY. cmd.exe reads .bat in the OEM codepage (cp950 here); UTF-8
REM CJK in comments gets mis-decoded and stray bytes execute as commands
REM (observed 2026-09-07: "'ck' is not recognized" + a bogus path error).
REM Rationale in Chinese lives in research/poc/conj_update.py, not here.
REM
REM Launched by research\ops\run_hidden.vbs (wscript has no console).
REM Proof it ran is the ARTIFACT: mtime and ok field of conj_clock_last.json.
REM Never LastTaskResult - an async launcher's code says nothing about the child.
setlocal
set ROOT=C:\Users\rfo\Desktop\flowbot\flow_system
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
echo [%date% %time%] conj_update start >> "%ROOT%\research\results\conj_update.log"
python "%ROOT%\research\poc\conj_update.py" >> "%ROOT%\research\results\conj_update.log" 2>&1
echo [%date% %time%] conj_update end rc=%errorlevel% >> "%ROOT%\research\results\conj_update.log"
endlocal
