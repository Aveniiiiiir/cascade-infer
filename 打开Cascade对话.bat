@echo off
chcp 65001 >nul
title Cascade 对话
start "Cascade Tunnel" /min ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -p 22094 -L 18080:127.0.0.1:8080 root@connect.nmb1.seetacloud.com
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:18080/chat.html
