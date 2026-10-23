@echo off
rem Builds bench\expert_trace.exe against the Tiel fork (llama-tiel\build).
rem Run it with the fork's bin dir (or its snapshot in bin\) on PATH.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\localllamacpp\bench
set F=D:\localllamacpp\llama-tiel
cl /nologo /O2 /EHsc /std:c++17 /utf-8 /I%F%\include /I%F%\ggml\include expert_trace.cpp ^
   /Fe:expert_trace.exe /Fo:%TEMP%\expert_trace.obj ^
   /link %F%\build\src\llama.lib %F%\build\ggml\src\ggml-base.lib %F%\build\ggml\src\ggml-cpu.lib
