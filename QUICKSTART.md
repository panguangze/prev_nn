# Quick Start Guide - foundry Workflow

This is a condensed guide to get you started quickly with the foundry workflow.

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] CUDA-capable GPU (optional but highly recommended)
- [ ] Git installed

## Installation (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/panguangze/prev_nn.git
cd prev_nn
```

### 2. Install Python Dependencies

```bash
pip install biopython numpy pandas scipy pyyaml freesasa
```

### 3. Install External Tools

```bash
# Create external directory
mkdir -p external

# Install RFdiffusion3
cd external
git clone <RFdiffusion3-repo-url> RFdiffusion3
cd RFdiffusion3
# Follow installation instructions from RFdiffusion3 repository
cd ../..

# Install ProteinMPNN
cd external
git clone https://github.com/dauparas/ProteinMPNN.git
cd ..

# Install RosettaFold3
cd external
git clone <RosettaFold3-repo-url> RosettaFold3
cd RosettaFold3
# Follow installation instructions from RosettaFold3 repository
cd ../..
```

### 4. Install GNU Parallel (optional but recommended)

```bash
# Using conda
conda install -c conda-forge parallel

# Or using apt (Ubuntu/Debian)
sudo apt-get install parallel
```

## Verify Installation

Run the compatibility checker:

```bash
python check_compatibility.py
```

This will verify that all required components are properly installed.

## Configuration (2 minutes)

### 1. Prepare Your Input Data

Place your PDB files in the `data/` directory:
- `data/3ckk.pdb` - Target protein (METTL1)
- `data/8d58.pdb` - Complex structure

### 2. Configure Parameters

Edit `config/params.yaml`:

```yaml
project:
  name: my_project
  batch_id: "batch_001"
  seed: 42

paths:
  rfdiffusion3_repo: "./external/RFdiffusion3"
  proteinmpnn: "./external/ProteinMPNN/protein_mpnn_run.py"
  rosettafold3_repo: "./external/RosettaFold3"
  
scale:
  rfdesigns_per_combo_per_lenbin: 10  # Start small for testing
  mpnn_num_seq_per_backbone_initial: 5
  
compute:
  gpus: [0]  # Adjust based on your GPU setup
  workers_per_gpu: 1
```

## Running the Pipeline (30 minutes - hours depending on scale)

### Full Pipeline

```bash
bash main.v100.sh
```

Or submit to SLURM:

```bash
sbatch main.v100.sh
```

### Individual Steps

For more control, run each step separately:

```bash
# Step 1: Prepare interface (1 minute)
python scripts/01_prepare_interface.py --params config/params.yaml

# Step 2: Select hotspots (30 seconds)
python scripts/02_select_hotspots.py --config config/params.yaml

# Step 3: Generate backbones with RFdiffusion3 (5-30 minutes)
bash scripts/03_run_rfdiffusion3.sh config/params.yaml

# Step 4: Design sequences with ProteinMPNN (2-10 minutes)
bash scripts/04_run_proteinmpnn.sh config/params.yaml

# Step 5: Predict structures with RosettaFold3 (10-60 minutes)
bash scripts/05_run_rf3.sh config/params.yaml

# Step 6: Rank designs (1 minute)
python scripts/06_rank_designs.py
```

## Expected Output

After completion, you'll find:

```
outputs/
├── targets/
│   ├── interface_candidates.json  # Interface analysis
│   └── hotspots_sets.json         # Selected hotspots
├── rfdiffusion3_raw/              # Generated backbones
├── mpnn_seqs/                     # Designed sequences
├── rf3_models/
│   └── predictions/               # Predicted structures
└── reports/
    └── ranked_designs.csv         # Final ranked results
```

## Quick Test Run (5 minutes)

Test with minimal parameters:

```bash
# Edit config for quick test
cat > config/params.test.yaml << 'EOF'
project:
  name: test
  batch_id: "test_001"
  seed: 42
  hotspot_count_grid: [3]
  length_bins:
    - {min: 60, max: 80}

paths:
  data_dir: "./data"
  work_dir: "./outputs_test"
  targets_dir: "./outputs_test/targets"
  rfdiffusion3_repo: "./external/RFdiffusion3"
  proteinmpnn: "./external/ProteinMPNN/protein_mpnn_run.py"
  rosettafold3_repo: "./external/RosettaFold3"
  mettl1_pdb: "./data/3ckk.pdb"
  complex_pdb: "./data/8d58.pdb"
  reference_complex_for_mask: "./data/8d58.pdb"

scale:
  rfdesigns_per_combo_per_lenbin: 2
  mpnn_num_seq_per_backbone_initial: 2

rfdd3:
  model_only_neighbors: true
  neighborhood_radius: 11.0
  inference_T: 50

rf3:
  with_template:
    initial:
      num_models: 1
      num_recycles: 3

compute:
  gpus: [0]
  workers_per_gpu: 1
  max_concurrent_rf3: 1
  max_concurrent_mpnn: 1
EOF

# Run test
python scripts/01_prepare_interface.py --params config/params.test.yaml
python scripts/02_select_hotspots.py --config config/params.test.yaml
bash scripts/03_run_rfdiffusion3.sh config/params.test.yaml
bash scripts/04_run_proteinmpnn.sh config/params.test.yaml
bash scripts/05_run_rf3.sh config/params.test.yaml
```

## Troubleshooting

### Common Issues

**Issue: "No GPU available"**
```bash
# Check GPU status
nvidia-smi

# Run on CPU (much slower)
export CUDA_VISIBLE_DEVICES=""
```

**Issue: "RFdiffusion3 not found"**
```bash
# Verify installation
ls -la external/RFdiffusion3/scripts/run_inference.py
```

**Issue: "Out of memory"**
```yaml
# Reduce GPU load in config/params.yaml
compute:
  workers_per_gpu: 1  # Reduce this
  min_free_mem_mb_for_gpu: 15000  # Increase this
```

## Next Steps

1. **Optimize Parameters**: Adjust `config/params.yaml` for your specific needs
2. **Scale Up**: Increase design numbers in `scale` section
3. **Read Full Documentation**: See [README.md](README.md) for comprehensive guide
4. **Analyze Results**: Review `outputs/reports/ranked_designs.csv`

## Performance Tips

- **Use Multiple GPUs**: Set `gpus: [0,1,2,3]` in config
- **Parallel Processing**: Increase `workers_per_gpu` if you have enough memory
- **Batch Processing**: Run multiple batches with different hotspot selections
- **Monitor Resources**: Use `nvidia-smi` to watch GPU usage

## Getting Help

- Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) if migrating from old workflow
- Review [CHANGELOG.md](CHANGELOG.md) for recent changes
- Run `python check_compatibility.py` to diagnose issues
- Check logs in `outputs/*/logs/` directories

## Example Workflow

Here's a complete example for a production run:

```bash
# 1. Setup
mkdir -p data external
# Place your PDB files in data/

# 2. Configure
cp config/params.yaml config/params.production.yaml
# Edit params.production.yaml with production parameters

# 3. Run
sbatch main.v100.sh

# 4. Monitor
tail -f outputs/rf3_models/log.txt

# 5. Analyze results
head -20 outputs/reports/ranked_designs.csv
```

## Resource Requirements

### Minimal Test
- 1 GPU with 8GB VRAM
- 16GB RAM
- ~5GB disk space
- ~30 minutes runtime

### Production Run
- 4 GPUs with 16GB+ VRAM each
- 64GB+ RAM
- ~500GB disk space
- Several hours runtime

## Support

For issues or questions:
1. Check existing documentation
2. Run compatibility checker
3. Review log files
4. Open an issue on GitHub

---

**Ready to start?** Run: `python check_compatibility.py`
