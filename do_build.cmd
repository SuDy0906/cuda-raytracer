@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%
Z:
echo === CMake Configure (Z:\, no spaces) ===
cmake -B Z:\build -S Z:\ -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
echo CONF_EXIT=%ERRORLEVEL%
if %ERRORLEVEL% NEQ 0 goto end
echo === Build ===
cmake --build Z:\build --config Release
echo BUILD_EXIT=%ERRORLEVEL%
:end
