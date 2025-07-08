# Script to update broken links after directory reorganization

# Define the path mappings for moved directories
$pathMappings = @{
    "quantum_ai_system/" = "ai/quantum/"
    "safety_ethics/" = "docs/guidelines/safety_ethics/"
    "safety_ethics.md" = "docs/guidelines/safety_ethics/README.md"
    "robotics/safety_ethics.md" = "docs/guidelines/safety_ethics/README.md"
    "temp_reorg/docs/robotics/safety_ethics.md" = "docs/guidelines/safety_ethics/README.md"
}

# Get all markdown files
$files = Get-ChildItem -Path . -Recurse -Include *.md | Where-Object { $_.FullName -notlike '*\node_modules\*' -and $_.FullName -notlike '*\.git\*' }

foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw
    $originalContent = $content
    
    # Replace each old path with the new one
    foreach ($mapping in $pathMappings.GetEnumerator()) {
        $content = $content -replace [regex]::Escape($mapping.Key), $mapping.Value
    }
    
    # Save the file if changes were made
    if ($content -ne $originalContent) {
        Write-Host "Updating links in $($file.FullName)"
        Set-Content -Path $file.FullName -Value $content -NoNewline
    }
}

Write-Host "Link updates complete!"
