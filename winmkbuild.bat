@echo off
setlocal enabledelayedexpansion

rem Windows counterpart of mkbuild.sh:
rem clean build dir, configure (Ninja) and build the project.

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "BUILD_DIR=%PROJECT_DIR%\build"

cd /d "%PROJECT_DIR%" || (echo Project not found & exit /b 1)

rem --- Detect CUDA architecture (sm_XX) from the GPU ------------------------
rem The header line (compute_cap) is filtered out by the regex below.
set "CUDA_ARCH=sm_89"
for /f "delims=" %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv 2^>nul') do (
    set "CC=%%i"
    echo "!CC!" | findstr /r "^[0-9][0-9]*\.[0-9][0-9]*$" >nul 2>nul
    if !errorlevel! equ 0 for /f "tokens=1,2 delims=." %%a in ("!CC!") do set "CUDA_ARCH=sm_%%a%%b"
)
echo CUDA architecture: %CUDA_ARCH%

rem --- Locate the Visual Studio developer environment -----------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VS_PATH=%%i"

if not defined VS_PATH (
    echo Visual Studio not found.
    exit /b 1
)

set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo C++ toolset not found at:
    echo   %VCVARS%
    exit /b 1
)

call "%VCVARS%"
where.exe cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Failed to set up the MSVC environment.
    exit /b 1
)

rem --- Clean, configure and build -------------------------------------------
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%" || exit /b 1
cd /d "%BUILD_DIR%" || exit /b 1

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCUDA_ARCH=%CUDA_ARCH%
if %errorlevel% neq 0 (
    echo CMake configure failed.
    exit /b 1
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    exit /b 1
)

cd /d "%PROJECT_DIR%"
echo Build finished: %BUILD_DIR%\bin\optix_app.exe
endlocal
