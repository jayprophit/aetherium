# Script to reorganize directories based on the new structure
# Run with: .\scripts\reorganize_dirs.ps1

$rootDir = Split-Path -Parent $PSScriptRoot
$moves = @(
    # Resources
    @{ From = "ai"; To = "resources\ai" },
    @{ From = "applications"; To = "resources\applications" },
    @{ From = "cross-platform"; To = "resources\cross-platform" },
    @{ From = "domains"; To = "resources\domains" },
    @{ From = "movement"; To = "resources\movement" },
    @{ From = "networking"; To = "resources\networking" },
    @{ From = "perception"; To = "resources\perception" },
    @{ From = "smart-devices"; To = "resources\smart-devices" },
    @{ From = "template resources"; To = "resources\templates" },
    
    # Development
    @{ From = "scripts"; To = "development\scripts" },
    @{ From = "tests"; To = "development\tests" },
    @{ From = "maintenance"; To = "development\maintenance" },
    @{ From = "performance"; To = "development\performance" },
    
    # Documentation
    @{ From = "guides"; To = "docs\guides" },
    @{ From = "contributing"; To = "docs\contributing" },
    @{ From = "governance"; To = "docs\governance" },
    
    # Systems
    @{ From = "platform"; To = "systems\platform" },
    @{ From = "platforms"; To = "systems\platforms" },
    @{ From = "protocols"; To = "systems\protocols" },
    @{ From = "core"; To = "systems\core" },
    
    # Operations
    @{ From = "iac"; To = "operations\iac" },
    @{ From = "process"; To = "operations\process" },
    
    # Special cases
    @{ From = "keras_project"; To = "examples\keras" },
    @{ From = "mcp"; To = "systems\protocols\mcp" }
)

# Create target directories if they don't exist
$moves | ForEach-Object {
    $targetDir = Join-Path $rootDir (Split-Path $_.To -Parent)
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        Write-Host "Created directory: $targetDir"
    }
}

# Move directories
$moves | ForEach-Object {
    $source = Join-Path $rootDir $_.From
    $destination = Join-Path $rootDir $_.To
    
    if (Test-Path $source) {
        if (Test-Path $destination) {
            Write-Warning "Destination already exists, skipping: $($_.To)"
        } else {
            Move-Item -Path $source -Destination (Split-Path $destination -Parent) -Force
            Rename-Item -Path (Join-Path (Split-Path $destination -Parent) (Split-Path $source -Leaf)) -NewName (Split-Path $destination -Leaf)
            Write-Host "Moved: $($_.From) -> $($_.To)"
        }
    } else {
        Write-Warning "Source not found: $($_.From)"
    }
}

Write-Host "\nReorganization complete!"
