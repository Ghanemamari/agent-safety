# run_extended_benchmark.ps1
# ===========================================================================
# Extended Benchmark Pipeline — Datasets OOD + Nouvelles Baselines
# ===========================================================================
# Utilisation:
#   .\run_extended_benchmark.ps1
#   .\run_extended_benchmark.ps1 -SkipDownload   # si datasets déjà téléchargés
#   .\run_extended_benchmark.ps1 -Load4Bit        # pour GPU limité
#   .\run_extended_benchmark.ps1 -QuickTest       # test rapide (1 dataset, 2 baselines)
#
# Résultats dans: data/results/  et  data/plots/

param(
    [switch]$SkipDownload,
    [switch]$Load4Bit,
    [switch]$QuickTest,
    [string]$Model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

$ErrorActionPreference = "Stop"
$python = ".\env\Scripts\python.exe"

# Vérifie que l'environnement Python existe
if (-not (Test-Path $python)) {
    Write-Host "[ERROR] Python env not found at $python" -ForegroundColor Red
    Write-Host "Run: python -m venv env && .\env\Scripts\pip install -r requirements.txt"
    exit 1
}

$load4bitFlag = if ($Load4Bit) { "--load4bit" } else { "" }

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ICLR AGENT SAFETY — Extended Pipeline" -ForegroundColor Cyan
Write-Host "  Model: $Model" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ── 0. Install / update dependencies ─────────────────────────────────────────
Write-Host "[0/6] Installing dependencies..." -ForegroundColor Yellow
& $python -m pip install -q -r requirements.txt

# ── 1. Download / prepare OOD datasets ───────────────────────────────────────
if (-not $SkipDownload) {
    Write-Host ""
    Write-Host "[1/6] Preparing datasets..." -ForegroundColor Yellow

    Write-Host "  Downloading InjecAgent (ACL 2024)..."
    & $python -m src.datasets.load_injecagent --out data/raw/injecagent.jsonl --download
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARNING] InjecAgent download failed. Skipping." -ForegroundColor DarkYellow
    }

    Write-Host "  Downloading AdvBench (Zou et al., 2023)..."
    & $python -m src.datasets.load_advbench `
        --out data/raw/advbench.jsonl `
        --download `
        --subset behaviors `
        --benign_jsonl data/raw/prompts_stealthy.jsonl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARNING] AdvBench download failed. Skipping." -ForegroundColor DarkYellow
    }

    Write-Host "  Preparing AgentDojo (requires: pip install agentdojo)..."
    & $python -m src.datasets.load_agentdojo --out data/raw/agentdojo.jsonl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARNING] AgentDojo not available. Install with: pip install agentdojo" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "[1/6] Skipping dataset download (-SkipDownload)" -ForegroundColor DarkGray
}

# ── 2. Feature extraction on all datasets ────────────────────────────────────
Write-Host ""
Write-Host "[2/6] Extracting hidden state features..." -ForegroundColor Yellow

$datasets = @(
    @{ name="stealthy";    jsonl="data/raw/prompts_stealthy_large.jsonl"; outdir="data/processed_stealthy" },
    @{ name="hard";        jsonl="data/raw/prompts_paired_hard_200.jsonl"; outdir="data/processed_hard" },
    @{ name="complex";     jsonl="data/raw/prompts_complex.jsonl";         outdir="data/processed_complex" }
)

if (-not $QuickTest) {
    $oodDatasets = @(
        @{ name="injecagent"; jsonl="data/raw/injecagent.jsonl";  outdir="data/processed_injecagent" },
        @{ name="advbench";   jsonl="data/raw/advbench.jsonl";    outdir="data/processed_advbench" }
    )
    $datasets += $oodDatasets
}

foreach ($ds in $datasets) {
    if (Test-Path $ds.jsonl) {
        Write-Host "  Extracting: $($ds.name)"
        mkdir $ds.outdir -Force | Out-Null
        $extractArgs = @("--model", $Model, "--input", $ds.jsonl, "--outdir", $ds.outdir)
        if ($Load4Bit) { $extractArgs += "--load4bit" }
        & $python -m src.run @extractArgs
    } else {
        Write-Host "  [SKIP] $($ds.jsonl) not found" -ForegroundColor DarkGray
    }
}

# ── 3. Linear Probe + MLP Ablation ───────────────────────────────────────────
Write-Host ""
Write-Host "[3/6] Running Linear Probe + MLP Ablation..." -ForegroundColor Yellow

$safeModel = $Model -replace "/", "_" -replace ":", "_"
$resultDir = "data/results"
mkdir $resultDir -Force | Out-Null

foreach ($ds in $datasets) {
    $npz = "$($ds.outdir)\${safeModel}_feats.npz"
    if (Test-Path $npz) {
        Write-Host "  $($ds.name): Linear Probe"
        & $python -m src.probes.train_linear_probe `
            --npz $npz `
            --out "$($ds.outdir)\${safeModel}_linear_metrics.json" `
            --sweep

        Write-Host "  $($ds.name): MLP Probe (ablation)"
        & $python -m src.baselines.mlp_probe `
            --npz $npz `
            --out "$($ds.outdir)\${safeModel}_mlp_metrics.json" `
            --compare_linear
    }
}

# ── 4. Surface Baselines (TF-IDF + Perplexity) ───────────────────────────────
Write-Host ""
Write-Host "[4/6] Running surface baselines (TF-IDF + Perplexity)..." -ForegroundColor Yellow

foreach ($ds in $datasets) {
    if (Test-Path $ds.jsonl) {
        Write-Host "  $($ds.name): TF-IDF"
        & $python -m src.baselines.text_tfidf `
            --input $ds.jsonl `
            --group_by_pair_id

        if (-not $QuickTest) {
            Write-Host "  $($ds.name): Perplexity"
            $pplArgs = @("-m", "src.baselines.statistical",
                         "--model", $Model,
                         "--input", $ds.jsonl,
                         "--out", "$($ds.outdir)\${safeModel}_ppl_metrics.json")
            if ($Load4Bit) { $pplArgs += "--load4bit" }
            & $python @pplArgs
        }
    }
}

# ── 5. Llama Guard Baseline ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Running Llama Guard baseline..." -ForegroundColor Yellow

if (-not $QuickTest) {
    $lgInputs = @(
        @{ ds="stealthy";    jsonl="data/raw/prompts_stealthy_large.jsonl" },
        @{ ds="injecagent";  jsonl="data/raw/injecagent.jsonl" }
    )
    foreach ($item in $lgInputs) {
        if (Test-Path $item.jsonl) {
            Write-Host "  Llama Guard on $($item.ds)..."
            $lgArgs = @("-m", "src.baselines.llama_guard",
                        "--input", $item.jsonl,
                        "--out", "data/results/llama_guard_$($item.ds)_metrics.json",
                        "--model", "meta-llama/Llama-Guard-3-1B")
            if ($Load4Bit) { $lgArgs += "--load4bit" }
            & $python @lgArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [WARNING] Llama Guard failed on $($item.ds). Access to meta-llama/Llama-Guard-3-1B required." -ForegroundColor DarkYellow
            }
        }
    }
} else {
    Write-Host "  [SKIP] Llama Guard (-QuickTest mode)" -ForegroundColor DarkGray
}

# ── 6. Generate Extended Plots ────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Generating publication plots..." -ForegroundColor Yellow

# Original plots
& $python scripts/generate_plots.py

# New extended plots
& $python scripts/generate_extended_plots.py `
    --plots heatmap ood ablation layers artifact

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Pipeline Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results  → data/results/"         -ForegroundColor White
Write-Host "Plots    → data/plots/"            -ForegroundColor White
Write-Host "Features → data/processed_*/"      -ForegroundColor White
Write-Host ""
Write-Host "Key files for the paper:" -ForegroundColor Cyan
Write-Host "  data/plots/auroc_heatmap.pdf           (Table 2 replacement)"
Write-Host "  data/plots/ood_generalization.pdf       (New Section 4.3)"
Write-Host "  data/plots/linear_vs_mlp_ablation.pdf   (New ablation study)"
Write-Host "  data/plots/artifact_analysis_token_length.pdf  (Addresses 1.00 AUROC concern)"
