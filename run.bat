@ECHO OFF
ping 127.0.0.1 -n 31 > nul
gunicorn run:app
