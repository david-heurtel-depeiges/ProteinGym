# Aggregating AMPLIFY Results into ProteinGym CSVs

## Overview - Complete Workflow

The aggregation process has **2 simple steps**:

1. **Generate config** - Recursively scans all model directories and auto-detects score columns  
2. **Run workflow** - Merges individual scores + computes performance metrics

Both steps are fully automated!

## Step 1: Generate Config (One Command)

```bash
cd /home/mila/d/david.heurtel-depeiges/ProteinGym/scripts/scoring_DMS_zero_shot

python generate_amplify_config.py \
  --score-folder /home/mila/d/david.heurtel-depeiges/scratch/proteingym/results/zero_shot_substitutions_scores \
  --output ../../config_thematic_lab.json
```

**What it does:**
- Recursively walks ALL subdirectories
- Finds folders with CSV score files  
- Auto-detects score column names (avg_score, mp_rank_00_model_states.pt, etc.)
- Ignores _analysis_cache* directories
- Creates hierarchical model names (e.g., AMPLIFY_2_120M_model_name)

## Step 2: Run Complete Workflow (One Command)

```bash
bash performance_AMPLIFY_substitutions.sh
```

**What it does:**
1. **MERGES** individual model CSVs into combined files (one per DMS, all models as columns)
2. **COMPUTES** 5 metrics: Spearman, AUC, MCC, NDCG, Top_recall  
3. **AGGREGATES** by DMS, protein, function, MSA depth, taxon, mutation depth

## Results

Performance metrics saved to: `scratch/proteingym/results/performance/all_models/`

Key files:
- **Summary_performance_DMS_substitutions_{metric}.csv** - Complete rankings with breakdowns
- **DMS_substitutions_{metric}_DMS_level.csv** - Per-assay performance  
- **DMS_substitutions_{metric}_Uniprot_level.csv** - Per-protein performance

## FAQ

### "Multiple models have the same input_score_name - is this normal?"

**YES!** This is expected. If different model versions output the same column name, they all get the same `input_score_name`. The `location` field distinguishes them.

### "Script skipped my models"

Check the score column detection. Run:
```bash
head -n 2 /path/to/your/model/some_DMS.csv
```

The score column should NOT be: mutant, mutated_sequence, DMS_score, DMS_score_bin

### Customizing paths

Edit `performance_AMPLIFY_substitutions.sh`:
- `INPUT_SCORES_BASE` - where your model scores are
- `CUSTOM_CONFIG` - your generated config file  
- `MERGED_SCORES_DIR` - where to save merged CSVs
- `OUTPUT_PERFORMANCE_FOLDER` - where to save metrics

## Summary

```bash
# Generate config (once)
python generate_amplify_config.py --score-folder <base_folder> --output config.json

# Run workflow (merges + computes metrics)
bash performance_AMPLIFY_substitutions.sh

# Done! Check results in scratch/proteingym/results/performance/
```
