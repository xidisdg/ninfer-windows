@echo off
setlocal EnableExtensions

rem NInfer Windows Release build (Ninja + MSVC + CUDA 13.1).
rem Usage:
rem   build.bat             configure (if needed) and build Release
rem   build.bat reconfigure delete build-windows-ninja and configure from scratch
rem Reconfigure is also possible by manually deleting the build-windows-ninja directory.

cd /d "%~dp0"

set "VSDEV=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
set "CMAKE=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "VCPKG_TOOLCHAIN=E:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake"
set "CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
set "BUILD_DIR=build-windows-ninja"

if not exist "%VSDEV%" echo error: VS 2026 dev command prompt not found at %VSDEV% & goto :error
if not exist "%CMAKE%" echo error: CMake not found at %CMAKE% & goto :error
if not exist "%NINJA%" echo error: Ninja not found at %NINJA% & goto :error
if not exist "%VCPKG_TOOLCHAIN%" echo error: vcpkg toolchain not found at %VCPKG_TOOLCHAIN% & goto :error
if not exist "%CUDA_COMPILER%" echo error: nvcc not found at %CUDA_COMPILER% & goto :error

call "%VSDEV%" -arch=x64 -host_arch=x64 || goto :error

if /i "%~1"=="reconfigure" (
  echo Removing %BUILD_DIR% ...
  rmdir /s /q "%BUILD_DIR%" || goto :error
)

if not exist "%BUILD_DIR%\build.ninja" (
  echo Configuring %BUILD_DIR% ...
  "%CMAKE%" -S . -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=120a ^
    -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler ^
    -DCMAKE_CUDA_COMPILER="%CUDA_COMPILER%" ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%" ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" || goto :error
)

echo Building %BUILD_DIR% (Release) ...
"%CMAKE%" --build %BUILD_DIR% -j || goto :error

echo.
echo Build OK:
echo   %BUILD_DIR%\apps\ninfer.exe
echo   %BUILD_DIR%\apps\ninfer-serve.exe
exit /b 0

:error
echo Build failed.
exit /b 1
