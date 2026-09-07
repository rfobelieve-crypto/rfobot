@echo off
REM Minute-level shadow detector for conjunction events (TODO 1.03).
REM Detector: research/poc/conj_watch.py  (shadow mode: logs, never signals)
REM Verified by three pre-registered controls before this task was created:
REM   conj_watch_parity  A arm 100.0%% (assembly)  B arm 92.6%% (thresholds)
REM   conj_watch_inject  88.9%% hit on real historical conjunctions
REM
REM Runs every minute. The point is to accumulate REAL end-to-end latency;
REM the 2-minute budget cannot be settled by an estimate.
REM
REM ASCII ONLY. cmd.exe reads .bat in the OEM codepage (cp950 here); UTF-8
REM CJK in comments gets mis-decoded and stray bytes execute as commands.
REM Rationale in Chinese lives in research/poc/make_conj_watch_bat.py.
REM
REM Launched by research\ops\run_hidden.vbs (wscript has no console).
REM Proof it ran is the ARTIFACT: mtime and ok field of conj_watch_last.json,
REM plus rows in conj_events_live. Never LastTaskResult - an async launcher's
REM exit code says nothing about the child.
setlocal
set ROOT=C:\Users\rfo\Desktop\flowbot\flow_system
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
python "%ROOT%\research\poc\conj_watch.py" >> "%ROOT%\research\results\conj_watch.log" 2>&1
endlocal
