# scripts/06_rank_designs_DEBUG_v2.py
import os
import json
import glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import freesasa
from tqdm import tqdm

try:
    import yaml
except ImportError:
    raise SystemExit("Please run: pip install pyyaml")

# --- 1. 配置加载 ---
PARAMS_FILE = "config/params.yaml"
if not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"配置文件 {PARAMS_FILE} 未找到。请确保路径正确。")

with open(PARAMS_FILE) as f:
    P = yaml.safe_load(f)

# --- 2. 路径设置 ---
report_dir = P["paths"]["reports_dir"]
os.makedirs(report_dir, exist_ok=True)

# --- 3. 参数加载 ---
FILT = P["filters"]["initial"]
WEI = P["ranking"]["weights"]
NORM = P["ranking"]["norm"]
REFERENCE_PDB = P["paths"]["reference_complex_for_mask"]

# --- 4. 全局对象初始化 ---
pdb_parser = PDBParser(QUIET=True)

# --- 辅助函数 (保持不变) ---
def get_chain_lengths(model_or_structure):
    return {chain.id: len(list(chain.get_residues())) for chain in model_or_structure}

def identify_target_and_binder_chains(model_object):
    chain_lengths = sorted(get_chain_lengths(model_object).items(), key=lambda item: item[1], reverse=True)
    if len(chain_lengths) < 2: return None, None
    target_chain_id, binder_chain_id = chain_lengths[0][0], chain_lengths[1][0]
    return model_object[target_chain_id], model_object[binder_chain_id]

def get_interface_residues(target_chain, binder_chain, cutoff=8.0):
    target_atoms = [atom for atom in target_chain.get_atoms() if atom.name == 'CA']
    binder_atoms = [atom for atom in binder_chain.get_atoms() if atom.name == 'CA']
    if not target_atoms or not binder_atoms: return [], []
    target_coords = np.array([atom.get_coord() for atom in target_atoms])
    binder_coords = np.array([atom.get_coord() for atom in binder_atoms])
    kdtree = cKDTree(binder_coords)
    indices = kdtree.query_ball_point(target_coords, r=cutoff)
    interface_target_residues = {target_atoms[i].get_parent() for i, atom_indices in enumerate(indices) if atom_indices}
    kdtree = cKDTree(target_coords)
    indices = kdtree.query_ball_point(binder_coords, r=cutoff)
    interface_binder_residues = {binder_atoms[i].get_parent() for i, atom_indices in enumerate(indices) if atom_indices}
    return list(interface_target_residues), list(interface_binder_residues)

def get_af2_scores(score_json_path):
    with open(score_json_path, 'r') as f: data = json.load(f)
    return data.get('iptm', 0.0), data.get('ptm', 0.0), data.get('plddt', [])

def calculate_interface_plddt(plddts, target_chain, binder_chain, binder_interface_residues):
    if not binder_interface_residues or not plddts: return 0.0
    offset = len(list(target_chain.get_residues()))
    all_binder_residues = list(binder_chain.get_residues())
    res_to_idx = {res: i for i, res in enumerate(all_binder_residues)}
    interface_plddt_scores = [plddts[offset + res_to_idx[res]] for res in binder_interface_residues if res in res_to_idx and (offset + res_to_idx[res]) < len(plddts)]
    return np.mean(interface_plddt_scores) if interface_plddt_scores else 0.0

def calculate_interface_pae(pae_json_path, target_len, binder_len):
    if not os.path.exists(pae_json_path): return float('inf')
    with open(pae_json_path, 'r') as f: data = json.load(f)
    pae_matrix = np.array(data.get('pae', []))
    if pae_matrix.size == 0: return float('inf')
    block1, block2 = pae_matrix[:target_len, target_len:], pae_matrix[target_len:, :target_len]
    if block1.size == 0 and block2.size == 0: return float('inf')
    return float((np.mean(block1) + np.mean(block2)) / 2)

def calculate_interface_bsa_optimized(structure, target_chain, binder_chain):
    try:
        complex_sasa = freesasa.calc(structure).totalArea()
        target_sasa = freesasa.calc(structure, freesasa.Structure.select(f'chain,{target_chain.id}')).totalArea()
        binder_sasa = freesasa.calc(structure, freesasa.Structure.select(f'chain,{binder_chain.id}')).totalArea()
        bsa = (target_sasa + binder_sasa - complex_sasa) / 2.0
        return bsa if bsa > 0 else 0.0
    except Exception: return 0.0

def calculate_clash_stats_optimized(target_chain, binder_chain):
    target_atoms = [a for a in target_chain.get_atoms() if a.element != 'H']
    binder_atoms = [a for a in binder_chain.get_atoms() if a.element != 'H']
    if not target_atoms or not binder_atoms: return (np.inf, np.inf)
    target_coords, binder_coords = np.array([a.get_coord() for a in target_atoms]), np.array([a.get_coord() for a in binder_atoms])
    tree = cKDTree(target_coords) if len(target_coords) < len(binder_coords) else cKDTree(binder_coords)
    coords_to_query = binder_coords if len(target_coords) < len(binder_coords) else target_coords
    distances, _ = tree.query(coords_to_query, k=1)
    if len(distances) == 0: return (np.inf, np.inf)
    return (float(np.percentile(distances, 5)), float(np.median(distances)))

def calculate_coverage_score(binder_chain, ref_mask_pts, cutoff=8.0):
    if ref_mask_pts.shape[0] == 0: return 0.0
    binder_cas = np.array([a.get_coord() for a in binder_chain.get_atoms() if a.name == 'CA'])
    if binder_cas.shape[0] == 0: return 0.0
    tree = cKDTree(binder_cas)
    covered_indices = tree.query_ball_point(ref_mask_pts, r=cutoff)
    covered_count = sum(1 for indices in covered_indices if indices)
    return covered_count / len(ref_mask_pts)

def get_bsa_threshold(length, stage="initial"):
    cfg = P["filters"][stage]["bsa_min_by_len"]
    if length < 80: return cfg["lt80"]
    if length < 100: return cfg["lt100"]
    return cfg["ge100"]

# --- 主逻辑 ---

# 1. 从参考复合物中提取WDR4界面残基 (保持不变)
print("正在从参考PDB生成界面掩码...")
structure = pdb_parser.get_structure("ref", REFERENCE_PDB)
target_model = next((model for model in structure if len(list(model.get_chains())) >= 2), None)
if target_model is None: raise SystemExit("错误：在参考PDB的所有模型中，都未能找到包含至少两条链的模型。")
ref_mettl1, ref_wdr4 = identify_target_and_binder_chains(target_model)
if not ref_mettl1 or not ref_wdr4: raise SystemExit("错误：识别靶点和binder链失败。")
_, wdr4_interface_residues = get_interface_residues(ref_mettl1, ref_wdr4)
ref_mask_points = np.array([res['CA'].get_coord() for res in wdr4_interface_residues if 'CA' in res])
print(f"成功生成参考界面掩码，包含 {len(ref_mask_points)} 个点。")


# --- DEBUGGING SETUP ---
debug_model_dirs = [
        "/scratch/project/cs_shuaicli/pgz/s2s/nncodes/test"
]

if not debug_model_dirs:
    print("\n错误：调试列表 `debug_model_dirs` 为空。")
    exit()

print(f"\n--- 进入调试模式，将只处理以下 {len(debug_model_dirs)} 个模型 ---")
for path in debug_model_dirs: print(f"- {path}")
print("--------------------------------------------------")

# 2. 遍历【指定的调试目录】并计算指标
all_results = []
for model_dir in tqdm(debug_model_dirs, desc="调试分析中"):
    if not os.path.isdir(model_dir):
        print(f"警告: 路径 '{model_dir}' 不是一个有效的目录，跳过。")
        continue

    design_name = os.path.basename(model_dir)

    score_json_list = glob.glob(os.path.join(model_dir, "*_scores_rank_001*.json"))
    if not score_json_list:
        print(f"调试信息: 在 {design_name} 中未找到 scores json 文件，跳过。")
        continue
    score_json = score_json_list[0]

    pdb_file_list = glob.glob(os.path.join(model_dir, "*_relaxed_rank_001*.pdb"))
    if not pdb_file_list:
        pdb_file_list = glob.glob(os.path.join(model_dir, "*_unrelaxed_rank_001*.pdb"))
    if not pdb_file_list:
        print(f"调试信息: 在 {design_name} 中未找到 PDB 文件，跳过。")
        continue
    pdb_file = pdb_file_list[0]

    # ######################################################################
    # --- 这是本次修正的核心 ---
    # 使用更宽松的模式来查找 PAE json 文件
    pae_json_list = glob.glob(os.path.join(model_dir, "*_predicted_aligned_error*.json"))
    # ######################################################################

    pae_json = pae_json_list[0] if pae_json_list else None
    if not pae_json:
        print(f"调试信息: 在 {design_name} 中未找到 PAE json 文件，PAE相关指标将为无效值。")

    print(f"\n正在处理: {design_name}")
    print(f"  - PDB: {os.path.basename(pdb_file)}")
    print(f"  - Score JSON: {os.path.basename(score_json)}")
    print(f"  - PAE JSON: {os.path.basename(pae_json) if pae_json else '未找到'}")

    try:
        structure_pred = pdb_parser.get_structure(design_name, pdb_file)
        model_pred = structure_pred[0]
        target_chain, binder_chain = identify_target_and_binder_chains(model_pred)
        if not target_chain or not binder_chain:
            print(f"调试信息: 在 {design_name} 的PDB中未能识别出两条链，跳过。")
            continue

        target_len, binder_len = len(list(target_chain.get_residues())), len(list(binder_chain.get_residues()))
        _, binder_interface_res = get_interface_residues(target_chain, binder_chain)
        iptm, _, plddts = get_af2_scores(score_json)

        pae_inter = calculate_interface_pae(pae_json, target_len, binder_len) if pae_json else float('inf')
        bsa = calculate_interface_bsa_optimized(structure_pred, target_chain, binder_chain)
        plddt_int = calculate_interface_plddt(plddts, target_chain, binder_chain, binder_interface_res)

        # ... (其余计算和数据记录部分保持不变) ...
        clash_p5, clash_median = calculate_clash_stats_optimized(target_chain, binder_chain)
        coverage = calculate_coverage_score(binder_chain, ref_mask_points)
        bsa_thr = get_bsa_threshold(binder_len, stage="initial")
        passed = (iptm >= FILT["iptm_min"] and pae_inter <= FILT["pae_inter_max"] and plddt_int >= FILT["plddt_interface_min"] and bsa >= bsa_thr and clash_p5 >= FILT["clash_p5_min"] and clash_median >= FILT["clash_median_min"] and coverage >= FILT["coverage_min"])
        inv_pae = 1.0 - min(pae_inter / NORM["pae_scale"], 1.0) if pae_inter != float('inf') else 0.0
        bsa_n = min(bsa / NORM["bsa_scale"], 1.0)
        score = (WEI["iptm"] * iptm + WEI["inv_pae_inter"] * inv_pae + WEI["bsa"] * bsa_n + WEI["plddt_interface"] * (plddt_int / 100.0) + WEI["coverage"] * coverage)
        all_results.append({'design_name': design_name, 'pdb_file': os.path.basename(pdb_file), 'passed_filter': passed, 'ranking_score': score, 'iptm': iptm, 'pae_inter': pae_inter, 'bsa': bsa, 'plddt_interface': plddt_int, 'coverage': coverage, 'clash_p5': clash_p5, 'clash_median': clash_median})
        print(f"  - 处理成功！iptm: {iptm:.4f}, pae_inter: {pae_inter:.4f}, bsa: {bsa:.2f}, score: {score:.4f}")

    except Exception as e:
        print(f"处理 {design_name} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()

# 3. 生成并保存报告 (保持不变)
if not all_results:
    print("\n--- 分析结束 ---")
    print("未能成功处理任何指定的模型，无法生成报告。请检查上面的调试信息。")
else:
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values(by=['passed_filter', 'ranking_score'], ascending=[False, False])
    output_csv = os.path.join(report_dir, "af2_ranked_designs_DEBUG.csv")
    df_sorted.to_csv(output_csv, index=False, float_format='%.4f')
    print("\n--- 分析完成 ---")
    print(df_sorted[['design_name', 'passed_filter', 'ranking_score', 'iptm', 'pae_inter', 'bsa']].head())
    print(f"\n[成功] 调试排名报告已写入: {output_csv}")
