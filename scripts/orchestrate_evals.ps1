$ErrorActionPreference = "Stop"

$Models = @(
    # @{ Name = "meta-llama/Llama-3.2-1B"; Alias = "llama3.2_1b"; Load4Bit = $false }, # DONE
    @{ Name = "meta-llama/Llama-3.2-3B"; Alias = "llama3.2_3b"; Load4Bit = $true },
    @{ Name = "Qwen/Qwen2.5-1.5B"; Alias = "qwen2.5_1.5b"; Load4Bit = $true },
    @{ Name = "mistralai/Mistral-7B-v0.3"; Alias = "mistral7b_v0.3"; Load4Bit = $true },
    @{ Name = "meta-llama/Meta-Llama-3.1-8B"; Alias = "llama3.1_8b"; Load4Bit = $true }
)

Write-Host "========================================="
Write-Host "STARTING UNIVERSAL ROBUSTNESS EVALUATION"
Write-Host "========================================="

foreach ($model in $Models) {
    $modelName = $model.Name
    $alias = $model.Alias
    $load4bit = $model.Load4Bit
    
    $outDir = "data/processed/$alias"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    
    $load4bitArg = ""
    if ($load4bit) {
        $load4bitArg = "--load4bit"
    }

    Write-Host "`n>>> Processing Model: $modelName ($alias) <<<"

    # 1. Extraction: Original files
    Write-Host "`n[1/7] Extracting original InjecAgent..."
    conda run -n gemma_spectral python -m src.extract.extract_activations --model $modelName --input data/raw/injecagent.jsonl --out $outDir/injecagent.npz $load4bitArg

    Write-Host "`n[2/7] Extracting original AdvBench..."
    conda run -n gemma_spectral python -m src.extract.extract_activations --model $modelName --input data/raw/advbench.jsonl --out $outDir/advbench.npz $load4bitArg

    Write-Host "`n[3/7] Extracting original AgentDojo..."
    conda run -n gemma_spectral python -m src.extract.extract_activations --model $modelName --input data/raw/agentdojo.jsonl --out $outDir/agentdojo.npz $load4bitArg

    # 2. Extraction: Ablated files
    Write-Host "`n[4/7] Extracting ablated InjecAgent..."
    conda run -n gemma_spectral python -m src.extract.extract_activations --model $modelName --input data/raw/injecagent_ablated.jsonl --out $outDir/injecagent_ablated.npz $load4bitArg

    Write-Host "`n[5/7] Extracting ablated AdvBench..."
    conda run -n gemma_spectral python -m src.extract.extract_activations --model $modelName --input data/raw/advbench_ablated.jsonl --out $outDir/advbench_ablated.npz $load4bitArg

    # 3. Post-Ablation Linear Probe AUROC
    Write-Host "`n[6/7] Running post-ablation probe on InjecAgent..."
    conda run -n gemma_spectral python -m src.probes.train_linear_probe --npz $outDir/injecagent_ablated.npz --out data/results/${alias}_ablation_probe.json
    
    # 4. Cross-Dataset Eval
    Write-Host "`n[7/7] Running cross-dataset evaluation..."
    conda run -n gemma_spectral python scripts/cross_dataset_eval.py --train $outDir/injecagent.npz $outDir/advbench.npz --test $outDir/agentdojo.npz --out data/results/${alias}_cross_eval.json
    
    # 5. Leave-One-Out Test
    Write-Host "`n[8/7] Running leave-one-out sensitivity analysis..."
    conda run -n gemma_spectral python scripts/leave_one_out.py --npz $outDir/injecagent.npz --out data/results/${alias}_leave_one_out.json

    Write-Host "`n>>> Completed $modelName <<<"
}

Write-Host "`n========================================="
Write-Host "EVALUATION COMPLETE FOR ALL MODELS"
Write-Host "========================================="
