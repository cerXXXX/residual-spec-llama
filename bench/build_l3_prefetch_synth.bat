@echo off
rem Builds bench\l3_prefetch_synth.exe and bench\l3_prefetch_trace.exe against the
rem Tiel fork's ggml (llama-tiel\build). Run them with the fork's bin dir (or its
rem snapshot in bin\) on PATH so ggml-base/ggml-cpu DLLs resolve.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\localllamacpp\bench
set F=D:\localllamacpp\llama-tiel
for %%s in (l3_prefetch_synth l3_prefetch_trace) do (
  cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 /utf-8 /I%F%\ggml\include %%s.cpp ^
     /Fe:%%s.exe /Fo:%TEMP%\%%s.obj ^
     /link %F%\build\ggml\src\ggml-base.lib %F%\build\ggml\src\ggml-cpu.lib Synchronization.lib
  if errorlevel 1 exit /b 1
)
