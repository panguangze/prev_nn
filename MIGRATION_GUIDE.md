# Migration Guide: Transitioning to foundry Workflow

This guide helps you migrate from the previous workflow (RFdiffusion + AlphaFold2) to the new foundry workflow (RFdiffusion3 + RosettaFold3).

## Overview of Changes

### Key Differences

| Component | Old Workflow | New Workflow | Change Type |
|-----------|-------------|--------------|-------------|
| Backbone Generation | RFdiffusion | RFdiffusion3 | Tool Upgrade |
| Sequence Design | ProteinMPNN | ProteinMPNN | No Change |
| Structure Prediction | AlphaFold2 Multimer | RosettaFold3 | Tool Replacement |
| Output Directory | `rfdiffusion_raw/` | `rfdiffusion3_raw/` | Path Change |
| Model Directory | `af2_models/` | `rf3_models/` | Path Change |

## Step-by-Step Migration

### 1. Update External Dependencies

#### Install RFdiffusion3
```bash
# Clone RFdiffusion3 from foundry repository
cd external/
git clone <RFdiffusion3-repo-url> RFdiffusion3
cd RFdiffusion3
# Follow installation instructions in RFdiffusion3 repository
```

#### Install RosettaFold3
```bash
# Clone RosettaFold3 from foundry repository
cd external/
git clone <RosettaFold3-repo-url> RosettaFold3
cd RosettaFold3
# Follow installation instructions in RosettaFold3 repository
```

### 2. Update Configuration Files

#### Configuration File Changes

**Old config structure:**
```yaml
paths:
  rfdiffusion_repo: "./external/RFdiffusion"
  alphafold_ref_pdb: "./data/8d58.pdb"

rfdd:
  model_only_neighbors: true
  neighborhood_radius: 11.0
  inference_T: 50
  noise_scale: 0.2

af2:
  with_template:
    initial:
      num_models: 2
      num_recycles: 3
      amber_relax: false

compute:
  max_concurrent_rf: 10
  max_concurrent_af2_initial: 1
```

**New config structure:**
```yaml
paths:
  rfdiffusion3_repo: "./external/RFdiffusion3"
  rosettafold3_repo: "./external/RosettaFold3"
  rosettafold3_ref_pdb: "./data/8d58.pdb"

rfdd3:
  model_only_neighbors: true
  neighborhood_radius: 11.0
  inference_T: 50
  noise_scale: 0.2
  diffusion_schedule: "linear"
  model_version: "v3"

rf3:
  with_template:
    initial:
      num_models: 2
      num_recycles: 3
      use_templates: true

compute:
  max_concurrent_rf3: 10
  max_concurrent_rf3_initial: 1
```

#### Automated Configuration Migration

Use this Python script to automatically migrate your config:

```python
#!/usr/bin/env python3
import yaml
import sys

def migrate_config(old_config_path, new_config_path):
    with open(old_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    if 'paths' in config:
        if 'rfdiffusion_repo' in config['paths']:
            config['paths']['rfdiffusion3_repo'] = config['paths']['rfdiffusion_repo'].replace('RFdiffusion', 'RFdiffusion3')
            del config['paths']['rfdiffusion_repo']
        
        if 'alphafold_ref_pdb' in config['paths']:
            config['paths']['rosettafold3_ref_pdb'] = config['paths']['alphafold_ref_pdb']
            del config['paths']['alphafold_ref_pdb']
        
        # Add new path
        config['paths']['rosettafold3_repo'] = './external/RosettaFold3'
    
    # Update rfdd to rfdd3
    if 'rfdd' in config:
        config['rfdd3'] = config['rfdd'].copy()
        config['rfdd3']['diffusion_schedule'] = 'linear'
        config['rfdd3']['model_version'] = 'v3'
        del config['rfdd']
    
    # Update af2 to rf3
    if 'af2' in config:
        config['rf3'] = {}
        for mode in ['with_template', 'no_template']:
            if mode in config['af2']:
                config['rf3'][mode] = {}
                for stage in ['initial', 'refine']:
                    if stage in config['af2'][mode]:
                        config['rf3'][mode][stage] = {
                            'num_models': config['af2'][mode][stage]['num_models'],
                            'num_recycles': config['af2'][mode][stage]['num_recycles'],
                            'use_templates': mode == 'with_template'
                        }
        del config['af2']
    
    # Update compute settings
    if 'compute' in config:
        if 'max_concurrent_rf' in config['compute']:
            config['compute']['max_concurrent_rf3'] = config['compute']['max_concurrent_rf']
            del config['compute']['max_concurrent_rf']
        
        if 'max_concurrent_af2_initial' in config['compute']:
            config['compute']['max_concurrent_rf3_initial'] = config['compute']['max_concurrent_af2_initial']
            del config['compute']['max_concurrent_af2_initial']
        
        if 'max_concurrent_af2_refine' in config['compute']:
            config['compute']['max_concurrent_rf3_refine'] = config['compute']['max_concurrent_af2_refine']
            del config['compute']['max_concurrent_af2_refine']
    
    # Update cleanup settings
    if 'cleanup' in config:
        if 'keep_af2_top_k_per_target' in config['cleanup']:
            config['cleanup']['keep_rf3_top_k_per_target'] = config['cleanup']['keep_af2_top_k_per_target']
            del config['cleanup']['keep_af2_top_k_per_target']
    
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Configuration migrated: {old_config_path} -> {new_config_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python migrate_config.py <old_config.yaml> <new_config.yaml>")
        sys.exit(1)
    
    migrate_config(sys.argv[1], sys.argv[2])
```

Save as `migrate_config.py` and run:
```bash
python migrate_config.py config/params.yaml.old config/params.yaml.migrated
```

### 3. Update Scripts and Workflows

#### Main Workflow Script

**Old (main.v100.sh):**
```bash
bash scripts/03_run_rfdiffusion.sh config/params.v100.yaml
bash scripts/05_run_af2_multimer.sh config/params.v100.yaml
```

**New (main.v100.sh):**
```bash
bash scripts/03_run_rfdiffusion3.sh config/params.v100.yaml
bash scripts/05_run_rf3.sh config/params.v100.yaml
```

### 4. Migrate Existing Data (Optional)

If you have existing output data and want to maintain directory structure consistency:

```bash
#!/bin/bash
# Script to migrate existing output directories

WORK_DIR="./outputs"

# Rename RFdiffusion output directory
if [ -d "$WORK_DIR/rfdiffusion_raw" ] && [ ! -d "$WORK_DIR/rfdiffusion3_raw" ]; then
    echo "Migrating rfdiffusion_raw to rfdiffusion3_raw..."
    mv "$WORK_DIR/rfdiffusion_raw" "$WORK_DIR/rfdiffusion3_raw"
fi

# Rename AlphaFold2 output directory
if [ -d "$WORK_DIR/af2_models" ] && [ ! -d "$WORK_DIR/rf3_models" ]; then
    echo "Migrating af2_models to rf3_models..."
    mv "$WORK_DIR/af2_models" "$WORK_DIR/rf3_models"
fi

echo "Migration complete!"
```

### 5. Test the New Workflow

Run a small test to verify the migration:

```bash
# Create a test configuration
cp config/params.yaml config/params.test.yaml

# Edit params.test.yaml to use smaller scale parameters:
# - rfdesigns_per_combo_per_lenbin: 5
# - mpnn_num_seq_per_backbone_initial: 2
# - hotspot_count_grid: [3]

# Run the test
bash main.v100.sh
```

## Compatibility Notes

### What Stays the Same

1. **Data preparation scripts** (`01_prepare_interface.py`, `02_select_hotspots.py`) - No changes
2. **ProteinMPNN integration** (`04_run_proteinmpnn.sh`) - Minor path update only
3. **Input data format** - PDB files and FASTA format unchanged
4. **Ranking script** (`06_rank_designs.py`) - Only output path updated

### What Changes

1. **RFdiffusion3 API** - Some parameters may have different names or defaults
2. **RosettaFold3 output format** - May differ from AlphaFold2 JSON structure
3. **GPU memory requirements** - RF3 may have different memory footprint than AF2
4. **Inference speed** - RF3 is generally faster than AF2

## Troubleshooting

### Common Migration Issues

#### Issue 1: "RFdiffusion3 not found"
**Solution:** Ensure RFdiffusion3 is properly installed and the path in config is correct:
```bash
ls -la external/RFdiffusion3/scripts/run_inference.py
```

#### Issue 2: "Invalid parameter for RFdiffusion3"
**Solution:** Check RFdiffusion3 documentation for updated parameter names. Common changes:
- Some diffusion parameters may have been renamed
- Check model version compatibility

#### Issue 3: "RosettaFold3 output format different"
**Solution:** Update ranking script to handle RF3 JSON format:
```python
# In 06_rank_designs.py, ensure proper parsing of RF3 outputs
def get_metrics_from_rf3_json(rf3_json):
    # RF3-specific parsing logic
    pass
```

#### Issue 4: "GPU memory issues with RosettaFold3"
**Solution:** Adjust worker settings in config:
```yaml
compute:
  workers_per_gpu: 1  # Reduce if OOM
  min_free_mem_mb_for_gpu: 15000  # Increase threshold
```

### Rollback Procedure

If you need to rollback to the old workflow:

```bash
# Restore old configuration
cp config/params.yaml.bk config/params.yaml

# Use old scripts
git checkout HEAD~1 -- scripts/03_run_rfdiffusion.sh
git checkout HEAD~1 -- scripts/05_run_af2_multimer.sh
git checkout HEAD~1 -- main.v100.sh
```

## Performance Comparison

Expected performance improvements with foundry workflow:

| Metric | Old (AF2) | New (RF3) | Improvement |
|--------|-----------|-----------|-------------|
| Inference Speed | ~10 min/target | ~3-5 min/target | 2-3x faster |
| GPU Memory | ~20GB | ~15GB | 25% less |
| Model Accuracy | Good | Better | Improved |
| Multi-chain | Good | Excellent | Enhanced |

## Best Practices

1. **Start Small**: Test with a subset of designs before full production run
2. **Monitor Resources**: Watch GPU memory and adjust `workers_per_gpu` accordingly
3. **Backup Configs**: Keep old configs for reference
4. **Version Control**: Track all configuration changes in git
5. **Documentation**: Document any custom modifications specific to your setup

## Getting Help

- Check the main [README.md](README.md) for comprehensive documentation
- Review RFdiffusion3 and RosettaFold3 official documentation
- Report issues on the GitHub repository

## Validation Checklist

Before considering migration complete:

- [ ] RFdiffusion3 installed and tested
- [ ] RosettaFold3 installed and tested
- [ ] All config files updated
- [ ] Test run completed successfully
- [ ] Output validation confirmed
- [ ] Performance benchmarks acceptable
- [ ] Team trained on new workflow
- [ ] Documentation updated

## Next Steps

After successful migration:

1. Run full-scale production runs
2. Compare results with old workflow for validation
3. Optimize parameters for your specific use case
4. Consider contributing improvements back to the repository

---

**Note:** This is a significant architectural change. Plan adequate time for testing and validation before deploying to production.
