conda activate westmarch

python -m spacy download en_core_web_sm

# List of Python modules to run
$modules = @(
    "src.extraction.01_preprocess_book"
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
