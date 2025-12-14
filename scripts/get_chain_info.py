# scripts/get_chain_info.py
# 保持原始逻辑
import sys
from Bio.PDB import PDBParser

if len(sys.argv) < 2:
    print("Usage: python get_chain_info.py <pdb_file>")
    sys.exit(1)

pdb_file = sys.argv[1]
p = PDBParser(QUIET=True)
s = p.get_structure("X", pdb_file)
chains = list(s.get_chains())
lens = []
for ch in chains:
    nres = sum(1 for r in ch.get_residues() if r.id[0] == " ")
    lens.append((ch.id, nres))
lens.sort(key=lambda x: x[1], reverse=True)
if len(lens) < 2:
    print("SKIP")
else:
    print(lens[0][0], lens[1][0])
