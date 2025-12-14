# scripts/clash_check.py
# 保持原始（轻量预筛）
import sys
from Bio.PDB import PDBParser
import numpy as np

def calculate_clash(pdb_file, clash_threshold=1.5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)
    chains = list(structure.get_chains())
    if len(chains) < 2:
        return 0
    target_atoms = [atom for res in chains[0].get_residues() for atom in res if atom.element != 'H']
    binder_atoms = [atom for res in chains[1].get_residues() for atom in res if atom.element != 'H']
    clash_count = 0
    for t_atom in target_atoms:
        t_coord = t_atom.coord
        for b_atom in binder_atoms:
            dist = np.linalg.norm(t_coord - b_atom.coord)
            if dist < clash_threshold:
                clash_count += 1
    return clash_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clash_check.py <pdb_file>")
        sys.exit(1)
    pdb_file = sys.argv[1]
    print(calculate_clash(pdb_file))
