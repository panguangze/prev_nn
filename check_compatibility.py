#!/usr/bin/env python3
"""
Compatibility checker for foundry workflow migration.

This script checks if your environment is ready for the new workflow
and identifies any potential issues before migration.
"""

import os
import sys
import subprocess
import json

class Color:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{text:^60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}\n")

def print_success(text):
    print(f"{Color.GREEN}✓{Color.RESET} {text}")

def print_warning(text):
    print(f"{Color.YELLOW}⚠{Color.RESET} {text}")

def print_error(text):
    print(f"{Color.RED}✗{Color.RESET} {text}")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"{description} not found: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print_success(f"{description}: {dirpath}")
        return True
    else:
        print_warning(f"{description} not found: {dirpath}")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        print_success(f"Python package '{package_name}' is installed")
        return True
    except ImportError:
        print_error(f"Python package '{package_name}' is NOT installed")
        return False

def check_command(command):
    """Check if a command is available"""
    try:
        subprocess.run([command, '--version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      check=False)
        print_success(f"Command '{command}' is available")
        return True
    except FileNotFoundError:
        print_warning(f"Command '{command}' is NOT available")
        return False

def check_yaml_config(filepath):
    """Check if YAML config is valid and has required fields"""
    try:
        import yaml
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for new required fields
        required_paths = [
            'paths.rfdiffusion3_repo',
            'paths.rosettafold3_repo',
            'paths.rosettafold3_ref_pdb'
        ]
        
        required_sections = ['rfdd3', 'rf3']
        
        issues = []
        
        for path in required_paths:
            parts = path.split('.')
            current = config
            for part in parts:
                if part not in current:
                    issues.append(f"Missing config: {path}")
                    break
                current = current[part]
        
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing section: {section}")
        
        if issues:
            for issue in issues:
                print_warning(issue)
            return False
        else:
            print_success(f"Configuration validated: {filepath}")
            return True
            
    except Exception as e:
        print_error(f"Error validating config: {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader'],
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True,
                              check=False)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print_success(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            return True
        else:
            print_warning("nvidia-smi command failed - GPUs might not be available")
            return False
    except FileNotFoundError:
        print_warning("nvidia-smi not found - running without GPU support")
        return False

def main():
    print_header("foundry Workflow Compatibility Checker")
    
    all_checks_passed = True
    
    # Check Python packages
    print_header("Checking Python Dependencies")
    packages = ['yaml', 'numpy', 'pandas', 'Bio', 'scipy', 'freesasa']
    for pkg in packages:
        if not check_python_package(pkg):
            all_checks_passed = False
    
    # Check external tools
    print_header("Checking External Tools")
    check_command('parallel')  # Optional but recommended
    
    # Check directory structure
    print_header("Checking Directory Structure")
    check_directory_exists('./data', 'Data directory')
    check_directory_exists('./scripts', 'Scripts directory')
    check_directory_exists('./config', 'Config directory')
    
    # Check essential files
    print_header("Checking Essential Files")
    essential_files = [
        ('config/params.yaml', 'Main configuration'),
        ('scripts/01_prepare_interface.py', 'Interface preparation script'),
        ('scripts/02_select_hotspots.py', 'Hotspot selection script'),
        ('scripts/03_run_rfdiffusion3.sh', 'RFdiffusion3 script'),
        ('scripts/04_run_proteinmpnn.sh', 'ProteinMPNN script'),
        ('scripts/05_run_rf3.sh', 'RosettaFold3 script'),
        ('scripts/06_rank_designs.py', 'Ranking script'),
        ('scripts/utils.py', 'Utility functions'),
    ]
    
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check new tool installations
    print_header("Checking Tool Installations")
    rfdiff3_exists = check_directory_exists('./external/RFdiffusion3', 
                                            'RFdiffusion3 repository')
    rf3_exists = check_directory_exists('./external/RosettaFold3', 
                                        'RosettaFold3 repository')
    
    if not rfdiff3_exists:
        print_error("  → Install with: git clone <RFdiffusion3-repo> external/RFdiffusion3")
        all_checks_passed = False
    
    if not rf3_exists:
        print_error("  → Install with: git clone <RosettaFold3-repo> external/RosettaFold3")
        all_checks_passed = False
    
    check_directory_exists('./external/ProteinMPNN', 'ProteinMPNN repository')
    
    # Check configuration
    print_header("Checking Configuration Files")
    if os.path.exists('config/params.yaml'):
        check_yaml_config('config/params.yaml')
    
    # Check GPU
    print_header("Checking GPU Availability")
    check_gpu()
    
    # Summary
    print_header("Summary")
    
    if all_checks_passed:
        print_success("All critical checks passed! ✓")
        print("\nYour environment appears ready for the foundry workflow.")
        print("\nNext steps:")
        print("  1. Verify RFdiffusion3 and RosettaFold3 are properly installed")
        print("  2. Update your configuration files if needed")
        print("  3. Run a small test: bash main.v100.sh")
        print("  4. Review MIGRATION_GUIDE.md for detailed instructions")
    else:
        print_warning("Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("  • Install missing Python packages: pip install -r requirements.txt")
        print("  • Clone missing repositories (see errors above)")
        print("  • Update configuration files (see MIGRATION_GUIDE.md)")
        print("  • Install GNU Parallel: conda install -c conda-forge parallel")
    
    print()
    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())
