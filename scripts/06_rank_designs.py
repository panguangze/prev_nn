# scripts/06_rank_designs.py
# 增强：加入界面遮挡率、clash-free分布、BSA分层阈值、两档过滤与加权排名
import os, json, glob, math
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
import freesasa
try:
    import yaml
except:
    raise SystemExit("Please pip install pyyaml")

PARAMS = "config/params.yaml"
with open(PARAMS) as f:
    P = yaml.safe_load(f)

pred_dir = os.path.join(P["paths"]["work_dir"], "rf3_models", "predictions")
report_dir = P["paths"]["reports_dir"]
os.makedirs(report_dir, exist_ok=True)

FILT = P["filters"]["initial"]  # 或根据阶段切换到 refine
WEI  = P["ranking"]["weights"]
NORM = P["ranking"]["norm"]

REFERENCE = P["paths"]["reference_complex_for_mask"]
# 从参考复合物中提取WDR4界面残基集合，用于遮挡率评估
parser = PDBParser(QUIET=True)
ref = parser.get_structure("R", REFERENCE)
ref_chains = list(ref.get_chains())
# 简化：取非METTL1链为“WDR4面”的接触残基集合
mettl1_chain_id = None
# 若有targets json则读
try:
    cand = json.load(open(os.path.join(P["paths"]["targets_dir"], "interface_candidates.json")))
    mettl1_chain_id = cand["mettl1_chain_id"]
except:
    mettl1_chain_id = ref_chains[0].id

def get_interface_mask(struct):
    # 返回目标界面点云（以WDR4接触面CA坐标）用于遮挡率估计
    chains = list(struct.get_chains())
    if len(chains) < 2:
        return np.zeros((0,3))
    # 选择METTL1链和另一条链
    ch_m = [ch for ch in chains if ch.id == mettl1_chain_id]
    if len(ch_m)==0: ch_m = [chains[0]]
    ch_m = ch_m[0]
    ch_o = [ch for ch in chains if ch.id != ch_m.id]
    if len(ch_o)==0: return np.zeros((0,3))
    ch_o = ch_o[0]
    # 界面：<8Å的残基CA
    from scipy.spatial import cKDTree
    A = np.array([a.coord for r in ch_m.get_residues() for a in r.get_atoms() if a.name=='CA'])
    B = np.array([a.coord for r in ch_o.get_residues() for a in r.get_atoms() if a.name=='CA'])
    if len(A)==0 or len(B)==0: return np.zeros((0,3))
    tree = cKDTree(A)
    mask_pts=[]
    for b in B:
        d, idx = tree.query(b, k=1)
        if d < 8.0:
            mask_pts.append(b)
    return np.array(mask_pts)

ref_mask = get_interface_mask(ref)

def get_metrics_from_json(rank_json):
    d = json.load(open(rank_json))
    iptm = d.get("iptm", d.get("iptm+ptm", 0.0))
    plddt = d.get("plddt", [])
    plddt_mean = np.mean(plddt) if isinstance(plddt, list) else (plddt if isinstance(plddt, (int,float)) else 0.0)
    return iptm, plddt_mean

def get_pae_from_json(pae_json):
    if not os.path.exists(pae_json):
        return float('inf')
    d = json.load(open(pae_json))
    # 简化：全均值
    return float(np.mean(d))

def interface_bsa(pdbfile):
    s = parser.get_structure("X", pdbfile)
    chains = list(s.get_chains())
    if len(chains) < 2:
        return 0.0
    ch1, ch2 = chains[0], chains[1]

    def sasa_of(struct):
        tmp = f"tmp_sasa_{os.getpid()}.pdb"
        io = PDBIO(); io.set_structure(struct); io.save(tmp)
        fs = freesasa.Structure(tmp)
        tot = freesasa.calc(fs).totalArea()
        os.remove(tmp)
        return tot

    # complex
    sc = sasa_of(s)
    class OneChain(Select):
        def __init__(self, cid): self.cid = cid
        def accept_chain(self, chain): return chain.id == self.cid
    io = PDBIO(); io.set_structure(s)
    tmpA = f"tmpA_{os.getpid()}.pdb"; tmpB=f"tmpB_{os.getpid()}.pdb"
    io.save(tmpA, OneChain(ch1.id)); io.save(tmpB, OneChain(ch2.id))
    sa = sasa_of(parser.get_structure("A", tmpA))
    sb = sasa_of(parser.get_structure("B", tmpB))
    os.remove(tmpA); os.remove(tmpB)
    return (sa + sb - sc) / 2.0

def plddt_interface_mean(rank_json, pdbfile):
    # 这里简化：直接用全局plddt均值代替界面plddt；可在后续细化
    iptm, plddt_mean = get_metrics_from_json(rank_json)
    return plddt_mean

def clash_stats(pdbfile):
    # 计算界面最近原子距离分布（简化）
    s = parser.get_structure("C", pdbfile)
    chains = list(s.get_chains())
    if len(chains) < 2: return (np.inf, np.inf)
    A = [a for r in chains[0].get_residues() for a in r.get_atoms() if a.element != 'H']
    B = [a for r in chains[1].get_residues() for a in r.get_atoms() if a.element != 'H']
    ds=[]
    for a in A:
      ac = a.coord
      mind = min(np.linalg.norm(ac - b.coord) for b in B)
      ds.append(mind)
    if len(ds)==0: return (np.inf, np.inf)
    return (float(np.percentile(ds, 5)), float(np.median(ds)))

def coverage_score(pdbfile, ref_mask_pts):
    # 估算遮挡率：binder的表面CA点（或全部CA）对ref_mask的近邻覆盖比例
    s = parser.get_structure("C", pdbfile)
    chains = list(s.get_chains())
    if len(chains) < 2 or len(ref_mask_pts)==0: return 0.0
    binder = chains[1]
    B = np.array([a.coord for r in binder.get_residues() for a in r.get_atoms() if a.name=='CA'])
    if len(B)==0: return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(B)
    covered=0
    for p in ref_mask_pts:
        d, idx = tree.query(p, k=1)
        if d < 8.0:  # 8Å 视为覆盖
            covered += 1
    return covered / len(ref_mask_pts)

def length_of_binder(pdbfile):
    s = parser.get_structure("L", pdbfile)
    chains = list(s.get_chains())
    if len(chains)<2: return 0
    binder = chains[1]
    L = len([r for r in binder.get_residues() if r.id[0]==' '])
    return L

def bsa_threshold_by_len(L, stage="initial"):
    cfg = P["filters"][stage]["bsa_min_by_len"]
    if L < 80:
        return cfg["lt80"]
    elif L < 100:
        return cfg["lt100"]
    else:
        return cfg["ge100"]

rows=[]
for rankjson in glob.glob(os.path.join(pred_dir, "**/*ranking_debug.json"), recursive=True):
    model_dir = os.path.dirname(rankjson)
    pae_jsons = glob.glob(os.path.join(model_dir, "*pae.json"))
    pdbs = glob.glob(os.path.join(model_dir, "*.pdb"))
    if not pdbs: continue
    pdbf = pdbs[0]
    iptm, plddt_mean = get_metrics_from_json(rankjson)
    paei = get_pae_from_json(pae_jsons[0]) if pae_jsons else float('inf')
    bsa = interface_bsa(pdbf)
    Lb = length_of_binder(pdbf)
    thr_bsa = bsa_threshold_by_len(Lb, stage="initial")
    plddt_int = plddt_interface_mean(rankjson, pdbf)
    p5, med = clash_stats(pdbf)
    cov = coverage_score(pdbf, ref_mask)

    passed = (iptm >= FILT["iptm_min"] and
              paei <= FILT["pae_inter_max"] and
              plddt_int >= FILT["plddt_interface_min"] and
              bsa >= thr_bsa and
              p5 >= FILT["clash_p5_min"] and
              med >= FILT["clash_median_min"] and
              cov >= FILT["coverage_min"])

    # 排名分数（归一化）
    inv_pae = 1.0 - min(paei / NORM["pae_scale"], 1.0)
    bsa_n = min(bsa / NORM["bsa_scale"], 1.0)
    score = (WEI["iptm"]*iptm + WEI["inv_pae_inter"]*inv_pae +
             WEI["bsa"]*bsa_n + WEI["plddt_interface"]*(plddt_int/100.0) +
             WEI["coverage"]*cov)

    rows.append(dict(
        model_dir=model_dir, pdb=pdbf,
        iptm=iptm, paei=paei, bsa=bsa, plddt_int=plddt_int,
        clash_p5=p5, clash_median=med, coverage=cov,
        binder_len=Lb, bsa_thr=thr_bsa,
        pass_initial=passed, score=score
    ))

df = pd.DataFrame(rows).sort_values(["pass_initial","score","bsa","iptm","coverage"], ascending=[False,False,False,False,False])
outcsv = os.path.join(report_dir, "af2_ranked_initial.csv")
df.to_csv(outcsv, index=False)
print(df.head(20))
print(f"[OK] Ranking written to {outcsv}")
