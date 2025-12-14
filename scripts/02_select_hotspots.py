# scripts/02_select_hotspots.py
import os, json, argparse, random, sys
from typing import List
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

try:
    from scipy.cluster.vq import kmeans2
except Exception as e:
    kmeans2 = None

def load_yaml(path: str):
    if not os.path.exists(path):
        print(f"[ERROR] 配置文件不存在: {path}", file=sys.stderr)
        sys.exit(1)
    if yaml is None:
        print("[ERROR] 需要 PyYAML，请先安装：python -m pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser(description="从候选界面残基中选取热点组合（由配置文件驱动）")
    p.add_argument("--config", default="config/params.yaml", help="YAML 配置文件路径")
    # 可选覆盖项（如提供，则覆盖配置文件对应值）
    p.add_argument("--candidates_json", default=None, help="候选残基 JSON（默认: paths.targets_dir/interface_candidates.json）")
    p.add_argument("--hotspot_counts", default=None, help="热点数量，逗号分隔（默认使用 project.hotspot_count_grid）")
    p.add_argument("--min_gap", type=int, default=None, help="同链序号最小间隔（默认 取 5 或配置覆盖）")
    p.add_argument("--max_sets_per_count", type=int, default=None, help="每种热点数生成的组合套数（默认 1 或配置覆盖）")
    p.add_argument("--num_clusters", type=int, default=4, help="kmeans 聚类簇数")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # 读取路径与默认参数
    targets_dir = cfg.get("paths", {}).get("targets_dir", "./outputs/targets")
    out_dir = targets_dir  # 按你现有流程，输出到 targets_dir
    os.makedirs(out_dir, exist_ok=True)

    # candidates_json 默认位置
    candidates_json = args.candidates_json or os.path.join(targets_dir, "interface_candidates.json")
    if not os.path.exists(candidates_json):
        print(f"[ERROR] 未找到 candidates_json: {candidates_json}", file=sys.stderr)
        sys.exit(1)

    # 随机种子：来自配置 project.seed（默认 42）
    seed = cfg.get("project", {}).get("seed", 42)
    random.seed(seed); np.random.seed(seed)

    # hotspot_counts：来自配置 project.hotspot_count_grid（例：[8, 12]）
    cfg_counts = cfg.get("project", {}).get("hotspot_count_grid", [8, 12])
    # CLI 覆盖（如传入 "--hotspot_counts 8"）
    if args.hotspot_counts:
        hotspot_counts = [int(x) for x in str(args.hotspot_counts).split(",") if x.strip()]
    else:
        hotspot_counts = list(map(int, cfg_counts))

    # 其他参数：提供合理默认，允许 CLI 覆盖
    min_gap = args.min_gap if args.min_gap is not None else 5
    max_sets_per_count = args.max_sets_per_count if args.max_sets_per_count is not None else 1
    num_clusters = args.num_clusters

    # 读取候选
    with open(candidates_json, "r") as f:
        cand = json.load(f)
    mettl1_chain_id = cand["mettl1_chain_id"]
    pool = cand["top_candidates"]

    # 偏好与关键残基（保持与原脚本一致，可改为从配置读取，如需我可以再扩展）
    key_residues = ["143", "179", "186", "264", "39", "40", "146", "147", "151", "182", "183"]
    preferred_types = {'LEU': 0.2, 'VAL': 0.2, 'PHE': 0.2, 'ARG': 0.15, 'LYS': 0.15, 'GLU': 0.15}

    for item in pool:
        item["resnum"] = item["resnum"].strip()
        try:
            item["resnum_int"] = int(item["resnum"])
        except Exception:
            item["resnum_int"] = 0
        item["preference_score"] = preferred_types.get(item.get("res_type", "UNK"), 0.0)
        item["key_bonus"] = 10.0 if item["resnum"] in key_residues else 0.0

    pool_sorted = sorted(
        pool,
        key=lambda x: (x.get("delta_sasa", 0.0) + x.get("contact_count", 0.0) + x["preference_score"] + x["key_bonus"]),
        reverse=True,
    )

    coords = np.array([it["coord"] for it in pool_sorted if "coord" in it and it["coord"] != [0.0, 0.0, 0.0]])
    idx_map = [i for i, it in enumerate(pool_sorted) if "coord" in it and it["coord"] != [0.0, 0.0, 0.0]]

    labels = None
    if kmeans2 is None:
        print("[WARN] 未安装 SciPy，将跳过空间聚类，仅使用序列间隔策略。建议安装：python -m pip install scipy")
    elif len(coords) >= num_clusters:
        _, labels = kmeans2(coords, num_clusters, minit="points")
    else:
        print("[WARN] 可用于 kmeans 的坐标不足，将回退到序列间隔策略。")

    def too_close(item, sel: List[dict]) -> bool:
        return any(abs(item["resnum_int"] - s["resnum_int"]) < min_gap for s in sel if item["resnum_int"] > 0 and s["resnum_int"] > 0)

    out_sets = []
    for hs in hotspot_counts:
        generated = 0
        attempts = 0
        while generated < max_sets_per_count and attempts < 50:
            attempts += 1
            selected = []
            # 强制包含部分关键残基（不超过四分之一）
            forced = [it for it in pool_sorted if it["resnum"] in key_residues][: max(1, hs // 4)]
            for it in forced:
                if not too_close(it, selected):
                    selected.append(it)

            # 按簇抽取
            if labels is not None:
                for c in range(num_clusters):
                    cluster_items = [pool_sorted[idx_map[i]] for i in range(len(idx_map)) if labels[i] == c and pool_sorted[idx_map[i]] not in selected]
                    cluster_items = [it for it in cluster_items if not too_close(it, selected)]
                    if cluster_items:
                        selected.append(cluster_items[0])
                    if len(selected) >= hs:
                        break

            # 回退补齐
            i = 0
            while len(selected) < hs and i < len(pool_sorted):
                it = pool_sorted[i]
                i += 1
                if it in selected:
                    continue
                if too_close(it, selected):
                    continue
                selected.append(it)

            if len(selected) == hs:
                out = {
                    "hotspots": [f"{mettl1_chain_id}:{it['resnum']}" for it in selected],
                    "hotspot_res_str": ",".join([f"{mettl1_chain_id}:{it['resnum']}" for it in selected]),
                    "mettl1_chain_id": mettl1_chain_id,
                    "hotspot_count": hs,
                    "min_gap": min_gap,
                }
                out_sets.append(out)
                generated += 1

    out_path = os.path.join(out_dir, "hotspots_sets.json")
    with open(out_path, "w") as w:
        json.dump(out_sets, w, indent=2)

    print(f"[OK] Generated {len(out_sets)} hotspot sets across counts {','.join(map(str, hotspot_counts))}.")
    print(f"[OK] Saved to {out_path}")

if __name__ == "__main__":
    main()
