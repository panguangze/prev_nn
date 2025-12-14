# scripts/01_prepare_interface.py
import os, json, argparse
from Bio.PDB import PDBIO
from utils import load_structure, best_chain_match, contact_pairs, sasa_by_chain, to_reskey, ChainSelect, residue_center
try:
    import yaml
except:
    raise SystemExit("Please pip install pyyaml")

parser = argparse.ArgumentParser()
parser.add_argument("--params", required=True, help="config/params.yaml")
args = parser.parse_args()

with open(args.params) as f:
    P = yaml.safe_load(f)

paths = P["paths"]
filters = P["filters"]
project = P["project"]

contact_cutoff = P.get("params",{}).get("contact_cutoff", None)  # 兼容老字段
if contact_cutoff is None:
    contact_cutoff = 5.0

topsasa_n = P.get("params",{}).get("topsasa_n", None)
if topsasa_n is None:
    topsasa_n = 50

os.makedirs(paths["targets_dir"], exist_ok=True)

s_m = load_structure(paths["mettl1_pdb"], "M")
s_c = load_structure(paths["complex_pdb"], "C")

chain_m_in_complex, ident = best_chain_match(s_m, s_c)
mettl1_chain_id = chain_m_in_complex.id

chains_complex = [ch for ch in s_c.get_chains()]
contact_counts = {}
for ch in chains_complex:
    if ch.id == mettl1_chain_id:
        continue
    pairs = contact_pairs(chain_m_in_complex, ch, cutoff=contact_cutoff)
    contact_counts[ch.id] = len(pairs)

wdr4_chain_id = max(contact_counts, key=lambda k: contact_counts[k])

# 计算ΔSASA
sasa_complex = sasa_by_chain(s_c)

io = PDBIO()
mettl1_only_path = os.path.join(paths["targets_dir"], f"8D58_{mettl1_chain_id}_only.pdb")
wdr4_only_path   = os.path.join(paths["targets_dir"], f"8D58_{wdr4_chain_id}_only.pdb")
mettl1_target_path = os.path.join(paths["targets_dir"], "mettl1_target.pdb")

io.set_structure(s_c); io.save(mettl1_only_path, ChainSelect([mettl1_chain_id]))
io.set_structure(s_c); io.save(wdr4_only_path,   ChainSelect([wdr4_chain_id]))
io.set_structure(s_c); io.save(mettl1_target_path, ChainSelect([mettl1_chain_id]))

from utils import load_structure as _ls
sasa_mono_m = sasa_by_chain(_ls(mettl1_only_path, "Mm"))
sasa_mono_w = sasa_by_chain(_ls(wdr4_only_path, "Ww"))

pairs = contact_pairs(chain_m_in_complex,
                      [ch for ch in chains_complex if ch.id==wdr4_chain_id][0],
                      cutoff=contact_cutoff)

mettl1_res_contact = {}
for resM, resW, d in pairs:
    keyM = to_reskey(resM)
    mettl1_res_contact[keyM] = mettl1_res_contact.get(keyM, 0) + 1

deltas = []
for (ch,resn), sasa_mono in sasa_mono_m.items():
    if ch != mettl1_chain_id: 
        continue
    sasa_c = sasa_complex.get((ch,resn), sasa_mono)
    delta = sasa_mono - sasa_c
    if delta > 0:
        deltas.append(((ch,resn), delta, mettl1_res_contact.get((ch,resn),0)))

deltas.sort(key=lambda x: (x[1], x[2]), reverse=True)
top_pool = deltas[:topsasa_n]

target_structure = _ls(mettl1_target_path, "T")
target_chain = list(target_structure.get_chains())[0]
target_len = len([res for res in target_chain.get_residues() if res.id[0]==' '])

res_dict = {str(res.id[1]).strip(): res for res in chain_m_in_complex.get_residues() if res.id[0]==' '}

def guess_ss(res):
    return "UNK"

out_json = {
  "project": project["name"],
  "batch_id": project["batch_id"],
  "mettl1_chain_id": mettl1_chain_id,
  "wdr4_chain_id": wdr4_chain_id,
  "contact_cutoff": contact_cutoff,
  "top_candidates": [
    {"chain":ch, "resnum":resn, "delta_sasa":float(dsasa), "contact_count":int(cc),
     "res_type": res_dict.get(resn.strip(), None).resname if res_dict.get(resn.strip()) else "UNK",
     "coord": [float(c) for c in residue_center(res_dict.get(resn.strip(), None))] if res_dict.get(resn.strip()) and residue_center(res_dict.get(resn.strip(), None)) is not None else [0.0, 0.0, 0.0],
     "ss": guess_ss(res_dict.get(resn.strip(), None)) }
    for (ch,resn), dsasa, cc in top_pool
  ],
  "mettl1_target_pdb": mettl1_target_path,
  "mettl1_target_len": target_len
}
with open(os.path.join(paths["targets_dir"], "interface_candidates.json"), "w") as f:
    json.dump(out_json, f, indent=2)

print(f"[OK] interface_candidates.json written. METTL1={mettl1_chain_id}, WDR4={wdr4_chain_id}, ident≈{ident:.2f}, topsasa_n={topsasa_n}")
