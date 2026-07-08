# 多种子节点取平均
import os
import json
import re
import logging
import random
import copy
import numpy as np
import torch
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# =========================
# 1. 配置区
# =========================
MODEL_NAME = "/home/ps/jiaq/InstructCom/code/qwen25-14b"
CHECKPOINT_PATH = "/home/ps/jiaq/InstructCom/output/synthetic-1000-20/nomod/final"

# 数据路径
BASE_PATH = "/home/ps/jiaq/InstructCom/dataset/synthetic-1000-20"
LABEL_FILE = "1000-20-node-labels.txt"
EDGE_FILE = "1000-20-hyperedges.txt"
OUTPUT_PATH = "/home/ps/jiaq/InstructCom/output/synthetic-1000-20/eval_results_nomod2.json"
LOG_FILE = "/home/ps/jiaq/InstructCom/output/synthetic-1000-20/debug_expansion_nomod2.log"

# =========================
# 强制指定运行的节点集 (替换之前的断点配置)
# =========================
# 将这里设为你想跑的社区，例如 "6"
FORCE_COMMUNITY = "6" 

# 把你剩下没跑完的节点组成一个列表直接填在这里
# 例如 333 是断点，你想跑它以及它后面的所有节点
FORCE_NODES = [333, 340]

# 推理参数
MAX_CANDIDATES = 6  # 基础候选节点数目
MAX_NEW_TOKENS = 1024
# MAX_NEW_TOKENS = 64
TEMPERATURE = 0.01
NUM_SEEDS_PER_COMM_TEST = 100  # 每个真实社区随机测试5个不同的种子
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

    def coarse_hypergraph_int_4_2hop_MC(
        self,
        node_1hop: list[int],
        hyperedges_1hop_dict: dict[int, list[int]],
        hyperedges_2hop_dict: dict[int, list[int]],
        K: int = 6,
        current_community: list[int] | None = None,
    ):
        # 1. 创建本地副本
        local_hyperedges_1hop = copy.deepcopy(hyperedges_1hop_dict)
        local_hyperedges_2hop = copy.deepcopy(hyperedges_2hop_dict)
        node_1hop = sorted(node_1hop)

        # ==========================================
        # 第一阶段：进行结构相似节点的合并（保留）
        # ==========================================
        # 2. 计算节点度（固定顺序遍历）
        degree_dict = {node: 0 for node in node_1hop}

        for _, hyperedge in sorted(local_hyperedges_1hop.items()):
            for node in hyperedge:
                if node in degree_dict:
                    degree_dict[node] += 1

        # 按度分组节点
        group_dict = defaultdict(list)
        for node, degree in sorted(degree_dict.items()):
            group_dict[degree].append(node)

        # 3. 计算超边权重
        community_nodes = set(current_community) if current_community else set()

        hyperedge_weights = {}
        for hyper_id, hyperedge in hyperedges_1hop_dict.items():
            hyperedge_count = len(hyperedge)
            community_node_count = sum(1 for node in hyperedge if node in community_nodes)

            if community_node_count > 0:
                weight = community_node_count / hyperedge_count
            else:
                weight = -1.0
            hyperedge_weights[hyper_id] = weight

        # 计算超边尺寸
        hyperedge_degree_map = {hid: len(he) for hid, he in hyperedges_1hop_dict.items()}

        # 构建节点到超边的映射
        node_to_hyperedges = defaultdict(list)
        for hyper_id, hyperedge in hyperedges_1hop_dict.items():
            for node in hyperedge:
                if node in degree_dict:
                    node_to_hyperedges[node].append(hyper_id)

        # 4. 计算节点相似度
        all_similarities = []

        for degree in sorted(group_dict.keys()):
            nodes_in_group = group_dict[degree]
            n_group = len(nodes_in_group)

            for i in range(n_group):
                for j in range(i + 1, n_group):
                    first_node = nodes_in_group[i]
                    second_node = nodes_in_group[j]

                    vec1_dict = defaultdict(list)
                    vec2_dict = defaultdict(list)

                    for hid in node_to_hyperedges.get(first_node, []):
                        hdeg = hyperedge_degree_map[hid]
                        vec1_dict[hdeg].append(hyperedge_weights[hid])

                    for hid in node_to_hyperedges.get(second_node, []):
                        hdeg = hyperedge_degree_map[hid]
                        vec2_dict[hdeg].append(hyperedge_weights[hid])

                    all_keys = sorted(set(vec1_dict.keys()) | set(vec2_dict.keys()))

                    vec1_parts = []
                    vec2_parts = []

                    for k in all_keys:
                        v1 = vec1_dict[k]
                        v2 = vec2_dict[k]
                        max_len = max(len(v1), len(v2))

                        v1_padded = sorted(v1 + [0.0] * (max_len - len(v1)))
                        v2_padded = sorted(v2 + [0.0] * (max_len - len(v2)))

                        vec1_parts.extend(v1_padded)
                        vec2_parts.extend(v2_padded)

                    if not all_keys:
                        vec1 = np.array([], dtype=float)
                        vec2 = np.array([], dtype=float)
                    else:
                        vec1 = np.array(vec1_parts, dtype=float)
                        vec2 = np.array(vec2_parts, dtype=float)

                    similarity = 0.0
                    if len(vec1) == 1 and len(vec2) == 1:
                        dist = np.linalg.norm(vec1 - vec2)
                        similarity = 1.0 / (1.0 + dist)
                    elif len(vec1) > 0 and len(vec2) > 0:
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)

                        if norm1 == 0 and norm2 == 0:
                            similarity = 1.0
                        elif norm1 == 0 or norm2 == 0:
                            similarity = 0.0
                        else:
                            similarity = max(0.0, np.dot(vec1, vec2) / (norm1 * norm2))

                    all_similarities.append({
                        'node_pair': (first_node, second_node),
                        'similarity': similarity,
                        'degree_group': degree
                    })

        # 5. 合并相似节点
        SIMILARITY_THRESHOLD = 0.95
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)

        merged_nodes_map = {}
        removed_nodes = set()
        merge_records = defaultdict(list)

        for sim in all_similarities:
            if sim['similarity'] < SIMILARITY_THRESHOLD:
                continue

            node1, node2 = sim['node_pair']

            if node1 in removed_nodes or node2 in removed_nodes:
                continue

            keep_idx = random.choice([0, 1])
            keep_node = sim['node_pair'][keep_idx]
            remove_node = sim['node_pair'][1 - keep_idx]

            merge_records[keep_node].append(remove_node)
            merged_nodes_map[remove_node] = keep_node
            removed_nodes.add(remove_node)

            for hyperedge in local_hyperedges_1hop.values():
                if remove_node in hyperedge:
                    hyperedge[:] = [x for x in hyperedge if x != remove_node]

            for hyperedge in local_hyperedges_2hop.values():
                if remove_node in hyperedge:
                    hyperedge[:] = [x for x in hyperedge if x != remove_node]

        # 6. 移除孤立节点
        for node in node_1hop:
            if node not in removed_nodes:
                connected = False
                for hyperedge in local_hyperedges_1hop.values():
                    if node in hyperedge:
                        connected = True
                        break

                if not connected:
                    removed_nodes.add(node)
                    for hyperedge in local_hyperedges_1hop.values():
                        if node in hyperedge:
                            hyperedge[:] = [x for x in hyperedge if x != node]
                    for hyperedge in local_hyperedges_2hop.values():
                        if node in hyperedge:
                            hyperedge[:] = [x for x in hyperedge if x != node]

        new_nodes = sorted(list(set(node_1hop) - removed_nodes))
        print(f"\n合并结果：共保留节点数 = {len(new_nodes)}，被合并节点数 = {len(removed_nodes)}")

        # 如果没有提供当前社区节点，直接按随机保留 (Fallback)
        if current_community is None or len(current_community) == 0:
            sample_size = min(K, len(new_nodes))
            fallback_nodes = random.sample(new_nodes, sample_size) if sample_size > 0 else []
            return fallback_nodes, merge_records, {"max_delta_m": 0.0, "positive_ratio": 0.0}

        # ==========================================
        # 模块度计算阶段（保留：用于输出 mod_stats 特征）
        # ==========================================
        community_nodes = set(current_community)
        all_hyperedges = {**local_hyperedges_1hop, **local_hyperedges_2hop}

        node_to_he = defaultdict(list)
        he_intersect_counts = {}

        e_in_base = 0.0
        e_out_base = 0.0

        for hid, he in all_hyperedges.items():
            he_len = len(he)
            if he_len == 0:
                continue

            for n in he:
                node_to_he[n].append(hid)

            count_c = sum(1 for n in he if n in community_nodes)
            he_intersect_counts[hid] = count_c

            if count_c > 0:
                e_in_base += count_c / he_len
                e_out_base += (he_len - count_c) / he_len

        base_hmc = e_in_base / e_out_base if e_out_base > 0 else 0.0
        node_scores = []

        for v in new_nodes:
            new_e_in = e_in_base
            new_e_out = e_out_base

            if v not in community_nodes:
                for hid in node_to_he.get(v, []):
                    he_len = len(all_hyperedges[hid])
                    count_c = he_intersect_counts[hid]

                    if count_c > 0:
                        new_e_in += 1.0 / he_len
                        new_e_out -= 1.0 / he_len
                    else:
                        new_e_in += 1.0 / he_len
                        new_e_out += (he_len - 1.0) / he_len

                new_hmc = new_e_in / new_e_out if new_e_out > 0 else float('inf')
                delta_m = new_hmc - base_hmc
            else:
                new_e_in_rem = e_in_base
                new_e_out_rem = e_out_base
                for hid in node_to_he.get(v, []):
                    he_len = len(all_hyperedges[hid])
                    count_c = he_intersect_counts[hid]

                    if count_c > 1:
                        new_e_in_rem -= 1.0 / he_len
                        new_e_out_rem += 1.0 / he_len
                    elif count_c == 1:
                        new_e_in_rem -= 1.0 / he_len
                        new_e_out_rem -= (he_len - 1.0) / he_len

                hmc_rem = new_e_in_rem / new_e_out_rem if new_e_out_rem > 0 else 0.0
                delta_m = base_hmc - hmc_rem

            node_scores.append({
                'node': v,
                'delta_m': delta_m
            })

        # 计算社区停止标志统计信息（保留提供给 Prompt 的统计指标）
        if not node_scores:
            mod_stats = {"max_delta_m": 0.0, "positive_ratio": 0.0}
        else:
            all_deltas = [record['delta_m'] for record in node_scores]
            max_delta_m = max(all_deltas)
            positive_count = sum(1 for d in all_deltas if d > 0)
            positive_ratio = positive_count / len(all_deltas)

            mod_stats = {
                "max_delta_m": round(max_delta_m, 4),
                "positive_ratio": round(positive_ratio, 3)
            }

        # ==========================================
        # 【核心修改点】改为随机筛选候选节点
        # ==========================================
        # 不再按照 delta_m 排序截取，直接从新生成的 new_nodes 节点池中随机抽取 K 个
        sample_size = min(K, len(new_nodes))
        keep_nodes = random.sample(new_nodes, sample_size) if sample_size > 0 else []
        print(f"随机筛选选中的候选节点: {keep_nodes}")

        keep_nodes_set = set(keep_nodes)
        removed_nodes = set(new_nodes) - keep_nodes_set

        # 6. 从超边中移除落选的节点
        for hyperedge in local_hyperedges_1hop.values():
            hyperedge[:] = [x for x in hyperedge if x not in removed_nodes]
        for hyperedge in local_hyperedges_2hop.values():
            hyperedge[:] = [x for x in hyperedge if x not in removed_nodes]

        print(f"\n随机筛选结果：共保留节点数 = {len(keep_nodes)}，因未被随机抽中而移除 = {len(removed_nodes)}")

        final_nodes = sorted(keep_nodes)

        # 7. 处理和过滤超边
        final_hyperedges = []
        for he in local_hyperedges_1hop.values():
            valid_he = [n for n in he if n not in removed_nodes]
            if len(valid_he) > 1:
                final_hyperedges.append(sorted(valid_he))

        # 直接返回结果
        return final_nodes, merge_records, mod_stats

    # 【修改点 1】：将 max_edges 改为动态参数，支持扩大候选集
    def get_subgraph_context(self, current_community, current_candidate):
        current_community_set = set(current_community)

        # ==========================================
        # 1. 获取一跳超边 (1-hop edges)
        # ==========================================
        hop1_edge_indices = set()
        for node in current_community:
            if node in self.node_to_edges:
                hop1_edge_indices.update(self.node_to_edges[node])

        # ==========================================
        # 2. 识别候选节点 (Candidates)
        #    用于寻找二跳边
        # ==========================================
        candidates = set()
        for idx in hop1_edge_indices:
            edge_nodes = self.hyperedges[idx]
            for node in edge_nodes:
                if node not in current_community_set:
                    candidates.add(node)
        print(f"候选节点:{candidates}")
        # ==========================================
        # 3. 获取二跳超边 (2-hop edges)
        # ==========================================
        hop2_edge_indices = set()
        for node in candidates:
            if node in self.node_to_edges:
                edge_indices = self.node_to_edges[node]
                hop2_edge_indices.update(edge_indices)

        # 剔除已经在一跳中包含的边，避免重复
        hop2_edge_indices = hop2_edge_indices - hop1_edge_indices

        # ==========================================
        # 4. 采样与合并 (Sampling Strategy)
        #    优先保留一跳边，剩余名额给二跳边
        # ==========================================
        total_edges_count = len(hop1_edge_indices) + len(hop2_edge_indices)
        neighbors = list(candidates)

        # 准备字典格式（为了兼容 utils 函数）
        hyperedges_1hop_dict = {idx: self.hyperedges[idx] for idx in hop1_edge_indices}
        hyperedges_2hop_dict = {idx: self.hyperedges[idx] for idx in hop2_edge_indices}
        # local_hypergraph = list(hyperedges_1hop_dict.values()) + list(hyperedges_2hop_dict.values())
        # print(f"局部图结构（包含两跳超边）：{local_hypergraph}")

        # 由于只需要给出一跳超边，所以缩减时只缩减一跳超边
        # 这里所返回的是加入后社区模块度提升最大的前5个节点，如果这五个节点正好在第一阶段节点合并中有合并的节点，那么当其被选中时，需要同时将merge_records当中记录的结构相同的其他节点也一起加入到当前社区当中。
        # 这里有一个问题就是如果这里加入的有错误节点是否需要生成训练数据？暂定不管结构相同的节点加入是否正确。
        final_node, merge_records, mod_stats = self.coarse_hypergraph_int_4_2hop_MC(
            neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, current_candidate, current_community)
        # print(f"缩减后的候选节点：{final_node}")
        # print(f"缩减后的一跳超边：{final_edge}")

        # 在这里增加候选节点关于统计信息的描述
        # 目前的潜在问题：给出的统计信息是完整的，超边未经过缩减的信息。给出的局部超图结构一跳超边是缩减过的，有可能对应不上。
        # 4. 格式化 Candidates 字符串
        # 格式示例: "Node 195: Shared hyperedges with community=2"
        candidates_stats = {}

        # 1. 预处理：计算每条超边包含的候选节点数量，用于快速判断是否与其他候选节点共享
        edge_candidate_counts = defaultdict(int)
        for node in final_node:
            if node in self.node_to_edges:
                for edge_idx in self.node_to_edges[node]:
                    edge_candidate_counts[edge_idx] += 1

        # 2. 统计每个节点的各项超边数据
        for node in final_node:
            hop1_count = 0
            hop2_count = 0
            shared_candidates_count = 0  # 新增：与其他候选节点共享的超边数目

            if node in self.node_to_edges:
                for edge_idx in self.node_to_edges[node]:
                    # 统计 1-hop 超边数量
                    if edge_idx in hop1_edge_indices:
                        hop1_count += 1

                    # 统计 2-hop 超边数量
                    if edge_idx in hop2_edge_indices:
                        hop2_count += 1

                    # 统计与其他候选节点共享的超边数量
                    # 如果该超边包含的候选节点数 > 1，说明它被当前节点和其他候选节点共享
                    if edge_candidate_counts[edge_idx] > 1:
                        shared_candidates_count += 1

            candidates_stats[node] = {
                "hop1": hop1_count,
                "hop2": hop2_count,
                "shared_candidates": shared_candidates_count  # 将新指标存入字典
            }

        # ==========================================
        # 6. 格式化 Candidates 字符串
        # ==========================================
        candidates_str_list = []

        for node, stats in candidates_stats.items():
            hop1_count = stats["hop1"]
            hop2_count = stats["hop2"]
            shared_cands = stats["shared_candidates"]

            # 在格式化字符串中追加 Shared with other candidates 的展示
            candidates_str_list.append(
                f"Node {node}: "
                f"Shared hyperedges with current community={hop1_count}, "
                f"External hyperedges={hop2_count}, "
                f"Shared hyperedges with other candidates={shared_cands}"
            )

        candidates_str = "\n".join(candidates_str_list)

        # print("Candidate statistics:")
        # print(candidates_str)

        return final_node, candidates_str, merge_records, mod_stats

    def get_community_state(self, current_community):
        """计算社区的当前状态，包括规模和平均内部度"""
        size = len(current_community)
        if size == 0:
            return {"size": 0, "avg_internal_degree": 0.0}

        comm_set = set(current_community)
        int_degrees = []

        for node in comm_set:
            d_int = sum(1 for eid in self.node_to_edges[node] if all(n in comm_set for n in self.hyperedges[eid]))
            int_degrees.append(d_int)

        avg_internal_degree = float(np.mean(int_degrees))

        return {
            "size": size,
            "avg_internal_degree": round(avg_internal_degree, 3)
        }

# =========================
# 3. 模型与推理工具
# =========================

# 【修改点 2】：全新的多节点 + STOP 提取器
def extract_nodes_from_output(text: str):
    """
    解析模型输出，返回一个字典:
    {"nodes": [id1, id2, ...], "stop": bool}
    """
    if not text:
        return {"nodes": [], "stop": False}

    # 1. 优先提取 Decision 块
    decision_match = re.search(r'Decision\s*:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    search_text = decision_match.group(1) if decision_match else text

    # 2. 判断是否包含 STOP
    if re.search(r'\b(?:STOP|REJECT_ALL)\b', search_text, re.IGNORECASE):
        return {"nodes": [], "stop": True}

    # 3. 提取 Node_xxx 格式的数字
    matches = re.findall(r'Node[_\s]*(\d+)', search_text, re.IGNORECASE)

    # 保底：如果没找到 Node_xxx，直接提数字
    if not matches:
        matches = re.findall(r'\d+', search_text)

    # 去重处理
    nodes = []
    seen = set()
    for m in matches:
        try:
            n = int(m)
            if n not in seen:
                seen.add(n)
                nodes.append(n)
        except:
            pass

    return {"nodes": nodes, "stop": False}


def build_prompt(tokenizer, current_community, community_stats, candidates_state_str):
    instruction = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."

    user_content = (
        f"{instruction}\n\n"
        f"### Current Community:\n{list(current_community)}\n\n"
        f"### Current Community state:\n{community_stats}\n\n"
        f"### Candidates Neighbors:\n{candidates_state_str}\n\n"
        # f"Please think step by step and provide your reasoning before making a Decision."
        f"Please think step by step and provide your reasoning. Finally, you must explicitly output your answer under the heading 'Decision:'."
        # f"Please directly output the ID of the node(s) to add or STOP. Do not provide any reasoning or explanation."
    )

    messages = [
        {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_metrics(pred_set, gt_set):
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    jaccard = intersection / union if union > 0 else 0
    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
    recall = intersection / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"jaccard": jaccard, "f1": f1, "precision": precision, "recall": recall}


# =========================
# 4. 主评估逻辑
# =========================

def evaluate():
    env = HypergraphEnvironment(BASE_PATH, LABEL_FILE, EDGE_FILE)

    logger.info("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    results = []
    total_metrics = defaultdict(float)

    with open("/home/ps/jiaq/InstructCom/dataset/synthetic-1000-20/synthetic-dataset_community_split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_labels = set(split_data["test"])

    # test_communities = [
    #     (label, nodes)
    #     for label, nodes in env.true_communities.items()
    #     if str(label) in test_labels
    # ]
    test_communities = [
    (label, nodes)
    for label, nodes in env.true_communities.items()
    # 同时满足：1. 在测试集中 2. 节点数量在指定区间内
        if str(label) in test_labels 
    # and 10 <= len(nodes) <= 300
    ]
    # test_communities = test_communities[:5]  # 测试前两个社区
    print(test_communities)
    logger.info(f"Starting evaluation on {len(test_communities)} communities...")

    # --- 将最大社区的范围限定在 involved_labels (即训练/测试/验证集涉及的社区) 当中 ---
     # 提取 JSON 中涉及的所有社区（比如 train, test, val 等全部合并）
    involved_labels = set()
    for labels_list in split_data.values():
        for lbl in labels_list:
            involved_labels.add(str(lbl))
    logger.info(f"Starting Baseline (Modularity-Only) evaluation on {len(test_communities)} communities...")

    max_comm_size                                                                                                                                                                                                                                                                                                                                                                                                 = max(
        len(nodes)
        for label, nodes in env.true_communities.items()
        if str(label) in involved_labels
    )

    for label, nodes in tqdm(test_communities):
        str_label = str(label)
        
        # --- 新增逻辑 1：如果开启了强制模式，且当前不是目标社区，直接跳过 ---
        if FORCE_COMMUNITY is not None and str_label != str(FORCE_COMMUNITY):
            continue

        gt_community = set(nodes)
        # --- 新增逻辑 2：替换节点集 ---
        if FORCE_COMMUNITY is not None and str_label == str(FORCE_COMMUNITY):
            # 强制使用你指定的节点集
            selected_seeds = FORCE_NODES
            actual_seed_count = len(selected_seeds)
            logger.info(f"Community {label}: 强制模式开启，只跑以下指定节点: {selected_seeds}")
        else:
            # （保留原有的正常随机采样逻辑，以防你以后把 FORCE_COMMUNITY 设为 None 时需要全量跑）
            actual_seed_count = min(NUM_SEEDS_PER_COMM_TEST, len(nodes))
            random.seed(42) 
            selected_seeds = random.sample(nodes, actual_seed_count)
            logger.info(f"Community {label}: Selected {actual_seed_count} seeds for testing: {selected_seeds}")
        # --- 【修改点 B】：为每个抽中的种子独立运行扩张逻辑 ---
        for seed_node in selected_seeds:
            logger.info(f"\n>>> Starting run for Community {label} with Seed {seed_node} <<<")
            
            # 状态重置（必须放在种子循环内部）
            current_community = {seed_node}
            log_steps = []
            stop_strikes = 0
            current_candidate = MAX_CANDIDATES
            step = 0
            
            # --- 以下保留你原有的 While True 核心推理逻辑 ---
            while len(current_community) < max_comm_size:
                logger.info(f"--- Step {step + 1} ---")
                step += 1
                # candidates列表记录当前候选节点id，candidates_state字符串记录候选节点的状态信息，merge_node字典记录合并节点信息，mod_stats字符串记录社区状态信息
                candidates, candidates_state, merge_node, mod_stats = env.get_subgraph_context(list(current_community), current_candidate)
                logger.info(f"Candidates count: {len(candidates)}")
                if not candidates:
                    log_steps.append("No candidates. Final Stop.")
                    break
                
                comm_state = env.get_community_state(current_community)
                comm_state["max_delta_m"] = mod_stats["max_delta_m"]
                comm_state["positive_ratio"] = mod_stats["positive_ratio"]
                # 构建提示符
                prompt = build_prompt(tokenizer, current_community, comm_state, candidates_state)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    # MAX_NEW_TOKENS控制模型最多生成多少个新 token,do_sample关闭随机采样，使用贪心解码
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=TEMPERATURE,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                    )
                full_response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                logger.info(f"current_community:{current_community}")
                logger.info(f"candidates:{candidates_state}")
                logger.info(f"Model Response: {full_response.strip()}")
                parsed_result = extract_nodes_from_output(full_response)
                pred_nodes = parsed_result["nodes"]
                is_stop = parsed_result["stop"]

                if is_stop:
                    break
                
                # 筛选有效节点，节点在候选节点集中，并且不在当前社区当中。
                valid_preds = [n for n in pred_nodes if n in candidates and n not in current_community]
                # 没有筛选出任何有效节点，扩大候选节点的数量
                if not valid_preds:
                    stop_strikes += 1
                    msg = f"No valid new nodes extracted from {pred_nodes} (Strike {stop_strikes})."
                    logger.info(msg)
                    if stop_strikes == 1:
                        current_candidate += 5
                        log_steps.append(msg + " Expanding context.")
                        continue
                    else:
                        log_steps.append(msg + " Final Stop.")
                        break
                
                # 加入有效节点的同时，加入和有效节点相似的节点
                for n in valid_preds:
                    current_community.add(n)
                    if n in merge_node:
                        current_community.update(merge_node[n])

                stop_strikes = 0
                current_candidate = MAX_CANDIDATES
                logger.info(f"Added nodes: {valid_preds}")
                log_steps.append(f"Added {valid_preds}")

            # --- 【修改点 C】：每个种子的运行结果独立记录 ---
            metrics = compute_metrics(current_community, gt_community)
            for k, v in metrics.items():
                total_metrics[k] += v  # 注意：这里是将所有(社区数 * 种子数)的结果加总求平均

            results.append({
                "community_label": label,    # 新增记录所属社区
                "seed": seed_node,
                "pred_community": list(current_community),
                "gt_community": list(gt_community),
                "metrics": metrics,
                "steps": len(log_steps)
            })

    avg_metrics = {k: v / len(results) for k, v in total_metrics.items() if len(results) > 0}
    print("\n" + "=" * 30)
    print("Final Evaluation Results:")
    print(f"Jaccard: {avg_metrics.get('jaccard', 0):.4f}")
    print(f"F1 Score: {avg_metrics.get('f1', 0):.4f}")
    print("=" * 30)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": avg_metrics, "details": results}, f, indent=2)


if __name__ == "__main__":
    evaluate()