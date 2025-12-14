# scripts/06_rank_designs_flat_dir.py
import os
import json
import glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import freesasa
from tqdm import tqdm
import argparse
import re
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("错误: PyYAML未安装。请运行: pip install pyyaml")

# ==============================================================================
# --- 0. 命令行参数解析 ---
# ==============================================================================
parser = argparse.ArgumentParser(
    description="对AlphaFold2输出的平铺目录中的设计进行分析、打分和排名。",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    "input_dir",
    type=str,
    help="包含所有AF2结果文件的平铺目录的路径。\n"
         "此目录中应直接包含所有设计的 .pdb, .json 等文件，没有子目录。"
)
args = parser.parse_args()

# ==============================================================================
# --- 1. 配置加载 ---
# ==============================================================================
PARAMS_FILE = "config/params.yaml"
if not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"配置文件 {PARAMS_FILE} 未找到。请确保脚本在项目根目录下运行。")

with open(PARAMS_FILE) as f:
    P = yaml.safe_load(f)

# ==============================================================================
# --- 2. 路径设置 ---
# ==============================================================================
# !!! 核心修改：输入目录由命令行参数指定 !!!
base_dir = args.input_dir
if not os.path.isdir(base_dir):
    raise NotADirectoryError(f"错误: 指定的输入路径 '{base_dir}' 不是一个有效的目录。")

report_dir = P["paths"]["reports_dir"]
os.makedirs(report_dir, exist_ok=True)

# ==============================================================================
# --- 3. 参数加载 ---
# ==============================================================================
FILT = P["filters"]["initial"]
WEI = P["ranking"]["weights"]
NORM = P["ranking"]["norm"]
REFERENCE_PDB = P["paths"]["reference_complex_for_mask"]

# ==============================================================================
# --- 4. 全局对象和辅助函数 (与原版相同) ---
# ==============================================================================
pdb_parser = PDBParser(QUIET=True)

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
    # 修正：确保链存在
    try:
        offset = len(list(target_chain.get_residues()))
        all_binder_residues = list(binder_chain.get_residues())
    except Exception:
        return 0.0
    res_to_idx = {res: i for i, res in enumerate(all_binder_residues)}
    interface_plddt_scores = [plddts[offset + res_to_idx[res]] for res in binder_interface_residues if res in res_to_idx and (offset + res_to_idx[res]) < len(plddts)]
    return np.mean(interface_plddt_scores) if interface_plddt_scores else 0.0

def calculate_interface_pae(pae_json_path, target_len, binder_len, design_name=""):
    if not os.path.exists(pae_json_path): return float('inf')
    with open(pae_json_path, 'r') as f: data = json.load(f)
    pae_data = data.get('pae') or data.get('predicted_aligned_error', [])
    pae_matrix = np.array(pae_data)
    if pae_matrix.size == 0:
        # print(f"  - 提示 ({design_name}): PAE JSON文件存在，但内部PAE列表为空。")
        return float('inf')
    total_len = target_len + binder_len
    if pae_matrix.shape[0] != total_len or pae_matrix.shape[1] != total_len:
        print(f"  - 警告 ({design_name}): PAE矩阵维度 ({pae_matrix.shape}) 与链长之和 ({total_len}) 不匹配。")
        return float('inf')
    block1, block2 = pae_matrix[:target_len, target_len:], pae_matrix[target_len:, :target_len]
    if block1.size == 0 or block2.size == 0: return float('inf')
    return float((np.mean(block1) + np.mean(block2)) / 2)


import freesasa
import os

def calculate_interface_bsa_optimized(pdb_file_path, target_chain, binder_chain):
    """
    【最终修正版 v5】
    根据错误信息 'float' object has no attribute 'totalArea' 进行最终修正。
    freesasa.selectArea 直接返回面积值（float），无需再调用 .totalArea()。
    这应该是与所有版本兼容的最终正确实现。
    """
    if not os.path.exists(pdb_file_path):
        print(f"  - BSA计算警告: PDB文件不存在于 '{pdb_file_path}'")
        return 0.0
    
    try:
        # 1. 创建 freesasa 结构对象
        structure = freesasa.Structure(pdb_file_path)
        
        # 2. 对整个复合物执行一次SASA计算
        result = freesasa.calc(structure)
        
        # 3. 获取复合物的总面积
        complex_sasa = result.totalArea()
        
        # 4. 定义选择字符串
        target_id = target_chain.id
        binder_id = binder_chain.id
        selection_strings = [
            f"target, chain {target_id}",
            f"binder, chain {binder_id}"
        ]
        
        # 5. 调用 freesasa.selectArea() 获取一个包含面积值的字典
        chain_areas = freesasa.selectArea(selection_strings, structure, result)
        
        # 6. 【核心修正】直接从字典中获取面积值（float），不再调用 .totalArea()
        target_sasa = chain_areas['target']
        binder_sasa = chain_areas['binder']
        
        # 7. 计算BSA
        bsa = (target_sasa + binder_sasa - complex_sasa) / 2.0
        
        return bsa if bsa > 0 else 0.0
        
    except Exception as e:
        # 捕获并打印任何freesasa计算中可能发生的错误
        design_name = os.path.basename(pdb_file_path).split('_unrelaxed')[0].split('_relaxed')[0]
        print(f"  - Freesasa计算错误 (文件: {design_name}): {e}")
        return 0.0

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

# ==============================================================================
# --- 主逻辑 ---
# ==============================================================================

# 1. 生成界面掩码 (与原版相同)
print("正在从参考PDB生成界面掩码...")
structure = pdb_parser.get_structure("ref", REFERENCE_PDB)
target_model = next((model for model in structure if len(list(model.get_chains())) >= 2), None)
if target_model is None: raise SystemExit("错误：参考PDB中未找到含多于一条链的模型。")
ref_mettl1, ref_wdr4 = identify_target_and_binder_chains(target_model)
if not ref_mettl1 or not ref_wdr4: raise SystemExit("错误：识别靶点和binder链失败。")
_, wdr4_interface_residues = get_interface_residues(ref_mettl1, ref_wdr4)
ref_mask_points = np.array([res['CA'].get_coord() for res in wdr4_interface_residues if 'CA' in res])
print(f"成功生成参考界面掩码，包含 {len(ref_mask_points)} 个点。")

# ==============================================================================
# --- 2. 核心修改：扫描平铺目录并识别唯一设计 ---
# ==============================================================================
print(f"\n正在扫描目录 '{base_dir}'...")
# 查找所有排名第一的PDB文件，作为识别设计的依据
all_rank1_pdbs = glob.glob(os.path.join(base_dir, "*_rank_001_*.pdb"))
if not all_rank1_pdbs:
    print(f"警告: 在 '{base_dir}' 中未找到任何 '*_rank_001_*.pdb' 文件。无法继续分析。")
    sys.exit()

# 从PDB文件名中提取唯一的设计前缀
unique_designs = set()
for pdb_path in all_rank1_pdbs:
    # 使用正则表达式从文件名中安全地提取前缀
    # 例如: '.../METTL1_design_10_..._sample_10_unrelaxed_rank_001_...pdb' -> 'METTL1_design_10_..._sample_10'
    base_name = os.path.basename(pdb_path)
    match = re.match(r"(.+?)_((un)?relaxed_)?rank_001", base_name)
    if match:
        unique_designs.add(match.group(1))

print(f"找到了 {len(unique_designs)} 个唯一的设计，开始分析...")

all_results = []
# --- 3. 核心修改：遍历唯一设计前缀，而不是目录 ---
for design_name in tqdm(sorted(list(unique_designs)), desc="分析所有设计中"):
    
    # --- 核心修改：使用设计前缀在平铺目录中精确查找文件 ---
    # 优先寻找 relaxed PDB，如果找不到再找 unrelaxed
    pdb_file = next(glob.iglob(os.path.join(base_dir, f"{design_name}_relaxed_rank_001*.pdb")), None)
    if not pdb_file:
        pdb_file = next(glob.iglob(os.path.join(base_dir, f"{design_name}_unrelaxed_rank_001*.pdb")), None)

    score_json = next(glob.iglob(os.path.join(base_dir, f"{design_name}_scores_rank_001*.json")), None)
    pae_json = next(glob.iglob(os.path.join(base_dir, f"{design_name}_predicted_aligned_error*.json")), None)

    # 如果核心文件不完整，则跳过此设计
    if not all([pdb_file, score_json]):
        # print(f"  - 警告 ({design_name}): 缺少PDB或Score JSON文件，已跳过。")
        continue
    
    try:
        structure_pred = pdb_parser.get_structure(design_name, pdb_file)
        model_pred = structure_pred[0]
        target_chain, binder_chain = identify_target_and_binder_chains(model_pred)
        if not target_chain or not binder_chain: continue
        
        target_len, binder_len = len(list(target_chain.get_residues())), len(list(binder_chain.get_residues()))
        _, binder_interface_res = get_interface_residues(target_chain, binder_chain)
        iptm, _, plddts = get_af2_scores(score_json)
        
        pae_inter = calculate_interface_pae(pae_json, target_len, binder_len, design_name) if pae_json else float('inf')
        bsa = calculate_interface_bsa_optimized(pdb_file, target_chain, binder_chain)
        plddt_int = calculate_interface_plddt(plddts, target_chain, binder_chain, binder_interface_res)
        clash_p5, clash_median = calculate_clash_stats_optimized(target_chain, binder_chain)
        coverage = calculate_coverage_score(binder_chain, ref_mask_points)
        
        bsa_thr = get_bsa_threshold(binder_len, stage="initial")
        passed = (iptm >= FILT["iptm_min"] and pae_inter <= FILT["pae_inter_max"] and plddt_int >= FILT["plddt_interface_min"] and bsa >= bsa_thr and clash_p5 >= FILT["clash_p5_min"] and clash_median >= FILT["clash_median_min"] and coverage >= FILT["coverage_min"])

        inv_pae = 1.0 - min(pae_inter / NORM["pae_scale"], 1.0) if pae_inter != float('inf') else 0.0
        bsa_n = min(bsa / NORM["bsa_scale"], 1.0)
        score = (WEI["iptm"] * iptm + WEI["inv_pae_inter"] * inv_pae + WEI["bsa"] * bsa_n + WEI["plddt_interface"] * (plddt_int / 100.0) + WEI["coverage"] * coverage)

        all_results.append({'design_name': design_name, 'pdb_file': os.path.basename(pdb_file), 'passed_filter': passed, 'ranking_score': score, 'iptm': iptm, 'pae_inter': pae_inter, 'bsa': bsa, 'plddt_interface': plddt_int, 'coverage': coverage, 'clash_p5': clash_p5, 'clash_median': clash_median, 'binder_len': binder_len, 'bsa_threshold': bsa_thr})

    except Exception as e:
        print(f"\n处理 {design_name} 时发生严重错误: {e}")

# ==============================================================================
# --- 4. 生成并保存报告 (与原版相同) ---
# ==============================================================================
if not all_results:
    print("\n--- 分析结束 ---")
    print("未能成功处理任何模型，无法生成报告。请检查错误信息或输入目录内容。")
else:
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values(by=['passed_filter', 'ranking_score', 'iptm', 'bsa', 'coverage'], ascending=[False, False, False, False, False])
    
    output_csv = os.path.join(report_dir, "af2_ranked_designs_flat.csv")
    df_sorted.to_csv(output_csv, index=False, float_format='%.4f')

    print("\n--- 分析完成 ---")
    print("报告预览 (前5名):")
    print(df_sorted[['design_name', 'passed_filter', 'ranking_score', 'iptm', 'pae_inter', 'bsa']].head())
    print(f"\n[成功] 最终排名报告已写入: {output_csv}")
