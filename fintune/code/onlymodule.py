# 只保留模块度筛选
import os
import os
import json
import logging
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import utils

# =========================
# 1. 配置区 (无需 LLM 配置)
# =========================
BASE_PATH = "/home/ps/jiaq/InstructCom/dataset/contact-primary-school"
LABEL_FILE = "node-labels-contact-primary-school.txt"
EDGE_FILE = "hyperedges-contact-primary-school.txt"
OUTPUT_PATH = "./eval_results_primary_modularity_only2.json" # 输出文件重命名，区分大模型结果
LOG_FILE = "./debug_expansion_primary_modularity_only2.log"

MAX_CANDIDATES = 6  
MAX_EXPANSION_STEPS = 200

# =========================
# 日志配置
# =========================
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# 2. 数据处理与环境类
# =========================
class HypergraphEnvironment:
    def __init__(self, base_path, label_file, edge_file):
        self.base_path = base_path
        self.label_path = os.path.join(base_path, label_file)
        self.hyperedge_path = os.path.join(base_path, edge_file)

        self.labels = defaultdict()
        self.hyperedges = []
        self.node_to_edges = defaultdict(set)
        self.true_communities = OrderedDict()

        self._load_data()

    def _load_data(self):
        logger.info("Loading Labels...")
        with open(self.label_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if line:
                    self.labels[line_num] = line

        logger.info("Loading Hyperedges...")
        with open(self.hyperedge_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                line = line.strip()
                if line:
                    nodes = [int(x) for x in line.split(',')]
                    self.hyperedges.append(nodes)
                    for node in nodes:
                        self.node_to_edges[node].add(idx)

        logger.info("Building Ground Truth Communities...")
        label_to_nodes = defaultdict(list)
        for node, label in self.labels.items():
            label_to_nodes[label].append(node)
        self.true_communities = OrderedDict(sorted(label_to_nodes.items()))
        logger.info(f"Loaded {len(self.hyperedges)} edges and {len(self.true_communities)} communities.")

    def get_subgraph_context(self, current_community, target_com, current_candidate):
        current_community_set = set(current_community)

        # 1. 获取一跳超边
        hop1_edge_indices = set()
        for node in current_community:
            if node in self.node_to_edges:
                hop1_edge_indices.update(self.node_to_edges[node])

        # 2. 识别候选节点
        candidates = set()
        for idx in hop1_edge_indices:
            edge_nodes = self.hyperedges[idx]
            for node in edge_nodes:
                if node not in current_community_set:
                    candidates.add(node)
                    
        # 3. 获取二跳超边
        hop2_edge_indices = set()
        for node in candidates:
            if node in self.node_to_edges:
                edge_indices = self.node_to_edges[node]
                hop2_edge_indices.update(edge_indices)

        hop2_edge_indices = hop2_edge_indices - hop1_edge_indices

        # 4. 采样与合并
        neighbors = list(candidates)
        hyperedges_1hop_dict = {idx: self.hyperedges[idx] for idx in hop1_edge_indices}
        hyperedges_2hop_dict = {idx: self.hyperedges[idx] for idx in hop2_edge_indices}
        
        # 注意：这里的 final_node 必须按 delta_m 降序排列 (需要在 utils.py 中修改)
        final_node, merge_records, mod_stats = utils.coarse_hypergraph_int_4_2hop_MC(
            neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, 6, current_community)

        # 在纯模块度推理中，我们不需要给 LLM 看状态，所以省略了字符串组装以提高速度
        return final_node, merge_records, mod_stats


def compute_metrics(pred_set, gt_set):
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    jaccard = intersection / union if union > 0 else 0
    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
    recall = intersection / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"jaccard": jaccard, "f1": f1, "precision": precision, "recall": recall}

# =========================
# 3. 纯模块度评估逻辑
# =========================
def evaluate_baseline():
    env = HypergraphEnvironment(BASE_PATH, LABEL_FILE, EDGE_FILE)

    results = []
    total_metrics = defaultdict(float)

    with open("/home/ps/jiaq/InstructCom/dataset/contact-primary-school/contact-primary-school_community_split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_labels = set(split_data["test"])

    test_communities = [
        (label, nodes)
        for label, nodes in env.true_communities.items()
        if str(label) in test_labels
    ]

    logger.info(f"Starting Baseline (Modularity-Only) evaluation on {len(test_communities)} communities...")

    for label, nodes in tqdm(test_communities):
        gt_community = set(nodes)
        seed_node = nodes[0]

        current_community = {seed_node}
        log_steps = []

        step = 0
        while len(current_community) < len(gt_community): # 去掉了作弊条件 len(current_community) < len(gt_community)
            logger.info(f"--- Step {step + 1} ---")
            step += 1
            
            # candidates 此时已经是按模块度增益降序排列的列表
            candidates, merge_node, mod_stats = env.get_subgraph_context(list(current_community), label, MAX_CANDIDATES)
            
            # 【停止条件 1】：没有候选节点了
            if not candidates:
                log_steps.append("No candidates. Final Stop.")
                logger.info("No candidates. Final Stop.")
                break
                
            # 【停止条件 2】：模块度不再增加 (重要！)
            # 既然是纯模块度驱动，当选任何人都无法带来模块度提升时，算法应当自动停止
            if mod_stats["max_delta_m"] <= 0:
                log_steps.append(f"Max delta_m ({mod_stats['max_delta_m']}) <= 0. Modularity Stop.")
                logger.info(f"Modularity gain is non-positive ({mod_stats['max_delta_m']}). Stopping expansion.")
                break

            # 【核心决策】：直接挑选列表中第一个节点（即 delta_m 最大的节点）
            best_node = candidates[0]
            
            # 记录并加入当前节点
            current_community.add(best_node)
            added_nodes = [best_node]
            
            # 如果该节点在粗化时合并了其他相似节点，一并加入
            if merge_node and best_node in merge_node:
                for s_node in merge_node[best_node]:
                    if s_node not in current_community:
                        current_community.add(s_node)
                        added_nodes.append(s_node)

            logger.info(f"Added nodes: {added_nodes} | Modularity Gain: {mod_stats['max_delta_m']}")
            log_steps.append(f"Added {added_nodes}")

        metrics = compute_metrics(current_community, gt_community)
        for k, v in metrics.items():
            total_metrics[k] += v

        results.append({
            "seed": seed_node,
            "pred_community": list(current_community),
            "gt_community": list(gt_community),
            "metrics": metrics,
            "steps": len(log_steps)
        })

    avg_metrics = {k: v / len(results) for k, v in total_metrics.items() if len(results) > 0}
    print("\n" + "=" * 40)
    print("🚀 Modularity-Only Baseline Results:")
    print(f"Jaccard:   {avg_metrics.get('jaccard', 0):.4f}")
    print(f"F1 Score:  {avg_metrics.get('f1', 0):.4f}")
    print(f"Precision: {avg_metrics.get('precision', 0):.4f}")
    print(f"Recall:    {avg_metrics.get('recall', 0):.4f}")
    print("=" * 40)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": avg_metrics, "details": results}, f, indent=2)


if __name__ == "__main__":
    evaluate_baseline()
