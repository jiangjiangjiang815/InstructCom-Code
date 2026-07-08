# 多种子节点取平均
import os
import json
import re
import logging
import random
import numpy as np
import torch
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import utils

# =========================
# 1. 配置区
# =========================
MODEL_NAME = "/home/ps/jiaq/InstructCom/code/qwen25-14b"
CHECKPOINT_PATH = "/home/ps/jiaq/InstructCom/output/contact-primary/nocomstate/final"

# 数据路径
BASE_PATH = "/home/ps/jiaq/InstructCom/dataset/contact-primary-school"
LABEL_FILE = "node-labels-contact-primary-school.txt"
EDGE_FILE = "hyperedges-contact-primary-school.txt"
OUTPUT_PATH = "/home/ps/jiaq/InstructCom/output/contact-primary/eval_results_nocom.json"
LOG_FILE = "/home/ps/jiaq/InstructCom/output/contact-primary/debug_expansion_nocom.log"

# 推理参数
MAX_CANDIDATES = 6  # 基础候选节点数目
MAX_NEW_TOKENS = 16
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

        # 构造 N-Set 语言编码的一跳局部超图
        edge_list_str = []
        for idx in hop1_edge_indices:
            edge_nodes = self.hyperedges[idx]
            nodes_str = ", ".join(f"v{node}" for node in edge_nodes)
            edge_list_str.append(f"({nodes_str})")
        
        local_hypergraph_str = ", ".join(edge_list_str)
        if not local_hypergraph_str:
            local_hypergraph_str = "No local hypergraph structure provided."

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
        final_node, merge_records, mod_stats = utils.coarse_hypergraph_int_4_2hop_MC(
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

        return final_node, candidates_str, merge_records, mod_stats, local_hypergraph_str

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


def build_prompt(tokenizer, current_community, candidates_state_str, local_hypergraph_str):
    instruction = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."

    user_content = (
        f"{instruction}\n\n"
        f"### Current Community Nodes:\n{list(current_community)}\n\n"
        f"### Candidate Neighbors:\n{candidates_state_str}\n\n"
        f"### Local Hypergraph Structure (1-hop):\n{local_hypergraph_str}\n\n"
        f"Please directly output the final decision without any reasoning or explanation."
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

def build_prompt(tokenizer, current_community, candidates_state_str, local_hypergraph_str, max_prompt_len=None):
    instruction = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."

    # 先拆成三段，但文本内容保持和原来完全一致
    prefix = (
        f"{instruction}\n\n"
        f"### Current Community Nodes:\n{list(current_community)}\n\n"
        f"### Candidate Neighbors:\n{candidates_state_str}\n\n"
        f"### Local Hypergraph Structure (1-hop):\n"
    )

    suffix = (
        f"\n\n"
        f"Please directly output the final decision without any reasoning or explanation."
    )

    # 如果没有长度限制，就保持原来的逻辑
    if max_prompt_len is None:
        user_content = prefix + local_hypergraph_str + suffix
        messages = [
            {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
            {"role": "user", "content": user_content}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 计算不放 local_hypergraph 时，固定 prompt 占多少 token
    empty_user_content = prefix + "" + suffix
    empty_messages = [
        {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
        {"role": "user", "content": empty_user_content}
    ]
    empty_prompt_text = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    empty_prompt_ids = tokenizer(empty_prompt_text, add_special_tokens=False)["input_ids"]

    # 剩余预算全部给 local_hypergraph
    graph_budget = max_prompt_len - len(empty_prompt_ids)

    if graph_budget <= 0:
        local_hypergraph_str = ""
    else:
        graph_ids = tokenizer(local_hypergraph_str, add_special_tokens=False)["input_ids"]
        if len(graph_ids) > graph_budget:
            graph_ids = graph_ids[:graph_budget]
            local_hypergraph_str = tokenizer.decode(graph_ids, skip_special_tokens=True)

    user_content = prefix + local_hypergraph_str + suffix

    messages = [
        {"role": "system", "content": "You are a Hypergraph Community Detection Expert."},
        {"role": "user", "content": user_content}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt_text
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

    with open("/home/ps/jiaq/InstructCom/dataset/contact-primary-school/contact-primary-school_community_split.json", "r", encoding="utf-8") as f:
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
        gt_community = set(nodes)
        
        # 如果社区节点总数少于配置的测试数，则测试所有节点；否则随机无放回抽取
        actual_seed_count = max(1, int(len(nodes) * 0.1))   # 10% 向下取整，至少1个
        # 或者如果希望上限不超过某个最大值，可结合 min 使用：
        # actual_seed_count = min(NUM_SEEDS_PER_COMM_TEST, max(1, int(len(nodes) * 0.1)))
        actual_seed_count = min(NUM_SEEDS_PER_COMM_TEST, len(nodes))
        # 固定随机种子（可选，为了实验可复现）
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
                candidates, candidates_state, merge_node, mod_stats, local_hypergraph_str = env.get_subgraph_context(list(current_community), current_candidate)
                logger.info(f"Candidates count: {len(candidates)}")
                if not candidates:
                    log_steps.append("No candidates. Final Stop.")
                    break
                
                # 1. 构建提示符并转为 Token IDs
                prompt_text = build_prompt(tokenizer, current_community, candidates_state, local_hypergraph_str)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                
                # 2. 复现训练集的特殊尾部保留截断法
                MAX_TOTAL_LENGTH = 2048
                max_prompt_len = MAX_TOTAL_LENGTH - MAX_NEW_TOKENS

                prompt_text = build_prompt(
                    tokenizer,
                    current_community,
                    candidates_state,
                    local_hypergraph_str,
                    max_prompt_len=max_prompt_len
                )

                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                
                # 3. 构造 Tensor 格式的输入
                input_ids_len = len(prompt_ids)
                inputs = {
                    "input_ids": torch.tensor([prompt_ids], dtype=torch.long).to(model.device),
                    "attention_mask": torch.tensor([[1] * input_ids_len], dtype=torch.long).to(model.device)
                }

                # 4. 模型推理
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=TEMPERATURE,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
                    )
                
                full_response = tokenizer.decode(output_ids[0][input_ids_len:], skip_special_tokens=True)

                logger.info(f"current_community:{current_community}")
                logger.info(f"Local Hypergraph: {local_hypergraph_str}") 
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