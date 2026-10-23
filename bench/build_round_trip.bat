@echo off
rem Builds bench\round_trip.exe (hypothesis No. 23 filter). Needs CUDA 12.6 + VS 2022.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\localllamacpp\bench
nvcc -O2 -arch=sm_86 -Xcompiler "/utf-8" round_trip.cu -o round_trip.exe -lcuda
