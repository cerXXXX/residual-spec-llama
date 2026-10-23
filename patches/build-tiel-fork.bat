@echo off
rem Builds the Tiel fork in llama-tiel, branch tiel, the same way as the earlier
rem own build: MSVC + CUDA 12.6, sm_86, Ninja, GGML_OPENMP=OFF. With OpenMP
rem the pinned-thread generation is broken and unreproducible, see CLAUDE.md.
rem Usage: patches\build-tiel-fork.bat [extra cmake --build args, e.g. -j 8]
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\localllamacpp\llama-tiel
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%
if exist build\CMakeCache.txt goto build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF
if errorlevel 1 exit /b 1
:build
cmake --build build --config Release %*
