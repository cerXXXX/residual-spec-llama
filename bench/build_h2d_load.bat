@echo off
rem Builds bench\h2d_load.exe (hypothesis No. 20: PCIe load vs CPU experts). Needs CUDA 12.6 + VS 2022.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\localllamacpp\bench
nvcc -O2 -arch=sm_86 -Xcompiler "/utf-8" h2d_load.cu -o h2d_load.exe
