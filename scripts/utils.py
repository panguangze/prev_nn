# scripts/utils.py
# 与你提供的版本一致，保持接口；仅确保依赖齐全
import os, json, math, numpy as np
from Bio.PDB import PDBParser, PPBuilder, PDBIO, Select
import freesasa
from scipy.spatial import KDTree

def load_structure(pdb_path, structure_id="S"):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, pdb_path)

def chain_seq(chain):
    ppb = PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(chain):
        seq += str(pp.get_sequence())
    return seq

def best_chain_match(struct_mettl1, struct_complex, min_identity=0.25):
    target_chain = list(struct_mettl1.get_chains())[0]
    seq_t = chain_seq(target_chain)
    best = (None, 0.0)
    from Bio import pairwise2
    for ch in struct_complex.get_chains():
        seq_c = chain_seq(ch)
        if len(seq_c) == 0 or len(seq_t) == 0:
            continue
        alns = pairwise2.align.globalxx(seq_t, seq_c, one_alignment_only=True, score_only=False)
        if not alns:
            continue
        aln = alns[0]
        identity = aln[2] / max(len(seq_t), len(seq_c))
        if identity > best[1]:
            best = (ch, identity)
    if best[0] is None or best[1] < min_identity:
        raise RuntimeError("无法在复合物中识别与 METTL1(3CKK) 匹配的链，请手动指定。")
    return best[0], best[1]

def residue_center(residue):
    coords = []
    for atom in residue.get_atoms():
        if atom.element != 'H':
            coords.append(atom.coord)
    if not coords:
        return None
    return np.mean(np.array(coords), axis=0)

def contact_pairs(chainA, chainB, cutoff=5.0):
    pairs = []
    atomsB = [a for res in chainB.get_residues() if res.id[0] == ' ' for a in res.get_atoms() if a.element != 'H']
    if len(atomsB)==0:
        return pairs
    coordsB = np.array([a.coord for a in atomsB])
    treeB = KDTree(coordsB)
    resB_list = [a.get_parent() for a in atomsB]
    for resA in chainA.get_residues():
        if resA.id[0] != ' ':
            continue
        atomsA = [a for a in resA.get_atoms() if a.element != 'H']
        if not atomsA: continue
        coordsA = np.array([a.coord for a in atomsA])
        min_d = float('inf'); closest_resB = None
        for coord in coordsA:
            dists, idxs = treeB.query(coord, k=1)
            if dists < min_d:
                min_d = dists; closest_resB = resB_list[idxs]
            if min_d < cutoff:
                break
        if min_d < cutoff:
            pairs.append((resA, closest_resB, min_d))
    return pairs

def sasa_by_chain(struct):
    io = PDBIO()
    tmp = "tmp_for_sasa.pdb"
    io.set_structure(struct)
    io.save(tmp)
    fs = freesasa.Structure(tmp)
    result = freesasa.calc(fs)
    res_sasa = {}
    for i in range(fs.nAtoms()):
        ch = fs.chainLabel(i)
        resn = fs.residueNumber(i)
        area = result.atomArea(i)
        key = (ch, resn)
        res_sasa[key] = res_sasa.get(key, 0.0) + area
    os.remove(tmp)
    return res_sasa

def to_reskey(res):
    ch = res.get_parent().id
    rn = str(res.id[1])
    return (ch, rn)

class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)
    def accept_chain(self, chain):
        return chain.id in self.chain_ids
