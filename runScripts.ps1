conda activate westmarch

# List of Python modules to run
$modules = @(
    "src.ingestion.ing_01_parse_raw_books"
)

foreach ($module in $modules) {
    Write-Host "Running Python module: $module"
    python -m $module

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Python module $module failed. Halting further execution."
        exit $LASTEXITCODE
    }
}

Write-Host "✅ All Python modules completed successfully."
