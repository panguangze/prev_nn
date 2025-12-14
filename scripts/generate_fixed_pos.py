# scripts/generate_fixed_pos.py
import sys
import json
from collections import defaultdict

def generate_fixed_positions(pdb_file, target_chain_id, binder_chain_id):
    """
    Parses a PDB file to create a fixed_positions JSON object.
    The target chain will have all its residues listed as fixed.
    The binder chain will have an empty list, meaning it's fully designable.
    """
    base_name = pdb_file.split('/')[-1].replace('.pdb', '')
    
    chain_residues = defaultdict(set)
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21]
                    res_num = int(line[22:26])
                    chain_residues[chain].add(res_num)
    except FileNotFoundError:
        sys.stderr.write(f"Error: PDB file not found at {pdb_file}\n")
        sys.exit(1)
    except ValueError:
        sys.stderr.write(f"Error: Could not parse residue number in {pdb_file}\n")
        sys.exit(1)

    # Build the final dictionary
    output_dict = {
        base_name: {
            # For the target chain, explicitly list all residue numbers to fix them
            target_chain_id: sorted(list(chain_residues.get(target_chain_id, []))),
            # For the binder chain, provide an empty list to make it fully designable
            binder_chain_id: []
        }
    }

    # Ensure the binder chain key exists even if it's not in the PDB (unlikely case)
    if binder_chain_id not in output_dict[base_name]:
        output_dict[base_name][binder_chain_id] = []

    print(json.dumps(output_dict))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python generate_fixed_pos.py <pdb_file> <target_chain_id> <binder_chain_id>\n")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    target_chain = sys.argv[2]
    binder_chain = sys.argv[3]
    
    generate_fixed_positions(pdb_path, target_chain, binder_chain)
