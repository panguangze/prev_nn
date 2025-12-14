#!/bin/bash
#SBATCH -p special_cs
#SBATCH -A pa_cs_department
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=cf2g-test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#python scripts/01_prepare_interface.py --params config/params.v100.yaml
python scripts/02_select_hotspots.py --config config/params.v100.yaml
bash scripts/03_run_rfdiffusion.sh config/params.v100.yaml
bash scripts/04_run_proteinmpnn.sh config/params.v100.yaml
bash scripts/05_run_af2_multimer.pwork.sh config/params.v100.yaml
