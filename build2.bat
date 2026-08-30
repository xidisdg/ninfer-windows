@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VSDEV=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set "CMAKE=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "VCPKG_TOOLCHAIN=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
set "CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
set "BUILD_DIR=build-windows-ninja2"

call "%VSDEV%" -arch=x64 -host_arch=x64 || goto :error

if not exist "%BUILD_DIR%\build.ninja" (
  "%CMAKE%" -S . -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=120a ^
    -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler ^
    -DCMAKE_CUDA_COMPILER="%CUDA_COMPILER%" ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" || goto :error
)

"%CMAKE%" --build %BUILD_DIR% -j 2>&1 || goto :error
exit /b 0

:error
echo Build failed.
exit /b 1
