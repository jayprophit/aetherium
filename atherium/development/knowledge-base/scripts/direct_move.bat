@echo off
echo Starting direct directory moves...

REM Create target directories if they don't exist
if not exist "resources\ai" mkdir "resources\ai"
if not exist "src\api" mkdir "src\api"
if not exist "resources\applications" mkdir "resources\applications"
if not exist "docs\contributing" mkdir "docs\contributing"
if not exist "systems\core" mkdir "systems\core"
if not exist "resources\cross-platform" mkdir "resources\cross-platform"
if not exist "resources\domains" mkdir "resources\domains"
if not exist "docs\governance" mkdir "docs\governance"
if not exist "docs\guides" mkdir "docs\guides"
if not exist "operations\iac" mkdir "operations\iac"
if not exist "examples\keras" mkdir "examples\keras"
if not exist "development\maintenance" mkdir "development\maintenance"
if not exist "systems\protocols\mcp" mkdir "systems\protocols\mcp"
if not exist "docs\meta" mkdir "docs\meta"
if not exist "resources\movement" mkdir "resources\movement"
if not exist "src\native" mkdir "src\native"
if not exist "resources\networking" mkdir "resources\networking"
if not exist "resources\perception" mkdir "resources\perception"
if not exist "development\performance" mkdir "development\performance"
if not exist "systems\platform" mkdir "systems\platform"
if not exist "systems\platforms" mkdir "systems\platforms"
if not exist "operations\process" mkdir "operations\process"
if not exist "systems\protocols" mkdir "systems\protocols"
if not exist "development\scripts" mkdir "development\scripts"
if not exist "resources\smart-devices" mkdir "resources\smart-devices"
if not exist "resources\templates" mkdir "resources\templates"
if not exist "development\tests" mkdir "development\tests"

echo Created target directories.

REM Move directories (using robocopy for reliability)
echo Moving directories...

if exist "ai" (
    echo Moving ai to resources\ai...
    robocopy "ai" "resources\ai" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "api" (
    echo Moving api to src\api...
    robocopy "api" "src\api" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "applications" (
    echo Moving applications to resources\applications...
    robocopy "applications" "resources\applications" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "contributing" (
    echo Moving contributing to docs\contributing...
    robocopy "contributing" "docs\contributing" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "core" (
    echo Moving core to systems\core...
    robocopy "core" "systems\core" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "cross-platform" (
    echo Moving cross-platform to resources\cross-platform...
    robocopy "cross-platform" "resources\cross-platform" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "domains" (
    echo Moving domains to resources\domains...
    robocopy "domains" "resources\domains" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "governance" (
    echo Moving governance to docs\governance...
    robocopy "governance" "docs\governance" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "guides" (
    echo Moving guides to docs\guides...
    robocopy "guides" "docs\guides" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "iac" (
    echo Moving iac to operations\iac...
    robocopy "iac" "operations\iac" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "keras_project" (
    echo Moving keras_project to examples\keras...
    robocopy "keras_project" "examples\keras" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "maintenance" (
    echo Moving maintenance to development\maintenance...
    robocopy "maintenance" "development\maintenance" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "mcp" (
    echo Moving mcp to systems\protocols\mcp...
    robocopy "mcp" "systems\protocols\mcp" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "meta" (
    echo Moving meta to docs\meta...
    robocopy "meta" "docs\meta" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "movement" (
    echo Moving movement to resources\movement...
    robocopy "movement" "resources\movement" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "native" (
    echo Moving native to src\native...
    robocopy "native" "src\native" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "networking" (
    echo Moving networking to resources\networking...
    robocopy "networking" "resources\networking" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "perception" (
    echo Moving perception to resources\perception...
    robocopy "perception" "resources\perception" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "performance" (
    echo Moving performance to development\performance...
    robocopy "performance" "development\performance" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "platform" (
    echo Moving platform to systems\platform...
    robocopy "platform" "systems\platform" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "platforms" (
    echo Moving platforms to systems\platforms...
    robocopy "platforms" "systems\platforms" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "process" (
    echo Moving process to operations\process...
    robocopy "process" "operations\process" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "protocols" (
    echo Moving protocols to systems\protocols...
    robocopy "protocols" "systems\protocols" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "scripts" (
    echo Moving scripts to development\scripts...
    robocopy "scripts" "development\scripts" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "smart-devices" (
    echo Moving smart-devices to resources\smart-devices...
    robocopy "smart-devices" "resources\smart-devices" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "template resources" (
    echo Moving template resources to resources\templates...
    robocopy "template resources" "resources\templates" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

if exist "tests" (
    echo Moving tests to development\tests...
    robocopy "tests" "development\tests" /E /MOVE /NFL /NDL /NJH /NJS /NC /NS /NP
)

echo Directory moves completed!

REM Clean up any empty directories
echo Cleaning up empty directories...
for /d %%d in (ai api applications contributing core cross-platform domains governance guides iac keras_project maintenance mcp meta movement native networking perception performance platform platforms process protocols scripts smart-devices "template resources" tests) do (
    if exist "%%d" (
        rmdir "%%d" 2>nul
        if errorlevel 1 echo Warning: Could not remove %%d - directory may not be empty
    )
)

echo Cleanup completed!
echo.
echo Current root directory structure:
dir /ad /b

echo.
echo Direct move operation completed!
