# Polypeptide Design Pipeline with foundry Workflow

This repository contains a computational pipeline for designing polypeptides targeting METTL1-WDR4 interface using the modern foundry workflow.

## Overview

The pipeline has been refactored to use the latest tools from the foundry repository:

### Workflow Components

1. **RFdiffusion3** - Advanced protein backbone generation (upgraded from RFdiffusion)
2. **ProteinMPNN** - Sequence design for generated backbones
3. **RosettaFold3** - Structure prediction and validation (upgraded from AlphaFold2 Multimer)

### Pipeline Stages

```
01_prepare_interface.py → 02_select_hotspots.py → 03_run_rfdiffusion3.sh → 04_run_proteinmpnn.sh → 05_run_rf3.sh → 06_rank_designs.py
```

#### Stage 1: Interface Preparation
- Script: `scripts/01_prepare_interface.py`
- Analyzes target and complex structures
- Identifies interface residues and hotspot candidates
- Outputs: `interface_candidates.json`

#### Stage 2: Hotspot Selection
- Script: `scripts/02_select_hotspots.py`
- Selects optimal hotspot combinations using spatial clustering
- Outputs: `hotspots_sets.json`

#### Stage 3: Backbone Generation (RFdiffusion3)
- Script: `scripts/03_run_rfdiffusion3.sh`
- Generates protein backbones using RFdiffusion3
- Supports custom diffusion schedules and model versions
- Parallel execution across multiple GPUs
- Outputs: `outputs/rfdiffusion3_raw/`

#### Stage 4: Sequence Design (ProteinMPNN)
- Script: `scripts/04_run_proteinmpnn.sh`
- Designs sequences for generated backbones
- Parallel processing using GNU Parallel
- Outputs: `outputs/mpnn_seqs/`

#### Stage 5: Structure Prediction (RosettaFold3)
- Script: `scripts/05_run_rf3.sh`
- Predicts structures using RosettaFold3
- Multi-GPU support with worker-per-GPU strategy
- Template-based prediction support
- Outputs: `outputs/rf3_models/predictions/`

#### Stage 6: Design Ranking
- Script: `scripts/06_rank_designs.py`
- Ranks designs based on multiple metrics
- Filters by iptm, pAE, pLDDT, BSA, and interface coverage
- Outputs: `outputs/reports/`

## Configuration

### Main Configuration File: `config/params.yaml`

Key sections:

#### Paths
```yaml
paths:
  rfdiffusion3_repo: "./external/RFdiffusion3"
  proteinmpnn: "./external/ProteinMPNN/protein_mpnn_run.py"
  rosettafold3_repo: "./external/RosettaFold3"
```

#### RFdiffusion3 Settings
```yaml
rfdd3:
  model_only_neighbors: true
  neighborhood_radius: 11.0
  inference_T: 50
  noise_scale: 0.2
  diffusion_schedule: "linear"
  model_version: "v3"
```

#### RosettaFold3 Settings
```yaml
rf3:
  with_template:
    initial:
      num_models: 2
      num_recycles: 3
      use_templates: true
    refine:
      num_models: 5
      num_recycles: 8
      use_templates: true
```

#### Compute Resources
```yaml
compute:
  gpus: [0,1,2,3]
  workers_per_gpu: 10
  max_concurrent_rf3: 10
  max_concurrent_mpnn: 10
  min_free_mem_mb_for_gpu: 12000
```

## Prerequisites

### Required Software
- Python 3.8+
- CUDA-capable GPUs (recommended)
- GNU Parallel (for ProteinMPNN stage)

### Python Dependencies
```bash
pip install biopython numpy pandas scipy pyyaml freesasa
```

### External Tools
1. **RFdiffusion3**: Clone from foundry repository
   ```bash
   git clone <RFdiffusion3-repo> external/RFdiffusion3
   ```

2. **ProteinMPNN**: 
   ```bash
   git clone https://github.com/dauparas/ProteinMPNN external/ProteinMPNN
   ```

3. **RosettaFold3**: Clone from foundry repository
   ```bash
   git clone <RosettaFold3-repo> external/RosettaFold3
   ```

## Usage

### Quick Start

1. Prepare your configuration:
   ```bash
   cp config/params.yaml config/params.custom.yaml
   # Edit params.custom.yaml with your settings
   ```

2. Run the full pipeline:
   ```bash
   bash main.v100.sh
   ```

   Or on SLURM:
   ```bash
   sbatch main.v100.sh
   ```

### Running Individual Stages

```bash
# Stage 1: Interface preparation
python scripts/01_prepare_interface.py --params config/params.yaml

# Stage 2: Hotspot selection
python scripts/02_select_hotspots.py --config config/params.yaml

# Stage 3: RFdiffusion3 backbone generation
bash scripts/03_run_rfdiffusion3.sh config/params.yaml

# Stage 4: ProteinMPNN sequence design
bash scripts/04_run_proteinmpnn.sh config/params.yaml

# Stage 5: RosettaFold3 structure prediction
bash scripts/05_run_rf3.sh config/params.yaml

# Stage 6: Design ranking
python scripts/06_rank_designs.py
```

## Key Improvements from Previous Version

### 1. RFdiffusion → RFdiffusion3
- Enhanced diffusion schedule options
- Improved model architecture (v3)
- Better control over inference parameters
- More stable training and inference

### 2. AlphaFold2 → RosettaFold3
- Faster inference times
- Better handling of protein complexes
- Improved accuracy for interface predictions
- Native support for templates
- More efficient GPU utilization

### 3. Architecture Improvements
- Better parallelization strategies
- Cleaner separation of concerns
- More configurable parameters
- Enhanced error handling and logging

## Output Structure

```
outputs/
├── targets/
│   ├── interface_candidates.json
│   ├── hotspots_sets.json
│   └── mettl1_target.pdb
├── rfdiffusion3_raw/
│   └── batch-*/
│       └── design_*.pdb
├── mpnn_seqs/
│   └── design_*/
│       └── seqs/*.fa
├── rf3_models/
│   ├── predictions/
│   │   └── design_*_sample_*/
│   ├── logs/
│   └── run/
└── reports/
    └── ranked_designs.csv
```

## Troubleshooting

### GPU Memory Issues
- Reduce `workers_per_gpu` in config
- Increase `min_free_mem_mb_for_gpu` threshold
- Reduce `num_models` or `num_recycles` for RF3

### Performance Optimization
- Adjust `max_concurrent_rf3` and `max_concurrent_mpnn`
- Use more GPUs if available
- Enable template caching for RF3

### Common Errors
1. **"No PDBs found"**: Check RFdiffusion3 output directory
2. **"GPU memory insufficient"**: Adjust worker settings
3. **"FASTA files not assembled"**: Verify ProteinMPNN output

## Contributing

When contributing, please:
1. Maintain backward compatibility where possible
2. Update documentation for new features
3. Add appropriate error handling
4. Follow existing code style

## License

[Specify license here]

## Citation

If you use this pipeline, please cite:
- RFdiffusion3 paper (foundry)
- RosettaFold3 paper (foundry)
- ProteinMPNN paper
- Original pipeline developers

## Contact

[Add contact information]

## Acknowledgments

This refactoring incorporates the advanced workflow from the foundry repository, bringing state-of-the-art tools for protein design to the prev_nn pipeline.
