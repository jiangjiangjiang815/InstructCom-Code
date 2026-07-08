# 2026.3.25 最新的一版数据生成逻辑
"""
    1、修改提示词，给出模糊定义
    2、修改数据生成逻辑，改成通过模块度选出正确节点，让大模型解释
"""
import sys
import argparse
import requests
import os.path
from collections import defaultdict, OrderedDict
import random
import json
import utils
import re
import numpy as np
import itertools

sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_CANDIDATE_TOP_K = 12

def parse_label_list(value):
    """Parse comma-separated labels from CLI input.

    If no value is provided, returns an empty set.
    """
    # 如果传入的值为空、None 或是空字符串
    if not value:
        return set()

    # 支持传入集合或列表直接转换，或者处理逗号分隔的字符串
    if isinstance(value, (set, list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}

    return {item.strip() for item in value.split(",") if item.strip()}


def positive_int(value):
    """Argparse type for positive integer arguments."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate hypergraph community-expansion instruction data."
    )
    parser.add_argument(
        "--base-path",
        default=os.path.join("datasets", "contact-primary-school"),
        help="Dataset directory containing labels.txt and hyperedge.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("datasets", "contact-primary-school"),
        help="Directory used to save generated JSONL files.",
    )
    parser.add_argument(
        "--data-output",
        default="primary_k12.json",
        help="Filename for expansion samples.",
    )
    parser.add_argument(
        "--stop-output",
        default="stop_primary_k12.json",
        help="Filename for STOP samples.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=positive_int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help="Number of candidate nodes kept after modularity filtering.",
    )
    # 将 default 设置为 None 或空字符串
    parser.add_argument(
        "--train_labels",
        type=str,
        default=None,  # 显式设为 None
        help="Comma-separated list of train labels (default: None, will use split cache)"
    )
    return parser.parse_args(argv)


def call_qwen_api(system_prompt, user_prompt):
    """调用通义千问 (Qwen) API"""
    # 阿里云官方 OpenAI 兼容接口地址
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # 请替换为你自己的 API Key
    }

    payload = {
        # 模型名称，例如 'qwen-max', 'qwen-plus', 'qwen-turbo'
        # 如果你想使用最新的推理模型，可以尝试 'qwen-max-latest'
        "model": "qwen-max",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,  # Qwen 的默认推荐通常在 0.7-1.0 之间
        "top_p": 0.8
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120,stream = True)
        res_json = response.json()

        # 诊断与容错处理
        if 'choices' not in res_json:
            # 阿里云的报错信息通常在 'message' 或 'code' 字段中
            error_msg = res_json.get('error', {}).get('message', '未知错误')
            print(f"⚠️ API 返回异常: {error_msg}")
            return None

        return res_json['choices'][0]['message']['content']

    except Exception as e:
        print(f"❌ 网络层错误: {e}")
        return None

def build_hypergraph_index(hyperedges):
    """
    构建倒排索引：节点 -> 所在的超边ID列表
    """
    node_to_edges = defaultdict(list)
    for edge_idx, nodes in enumerate(hyperedges):
        for node in nodes:
            node_to_edges[node].append(edge_idx)
    return node_to_edges


def get_subgraph_context(
        current_community,
        node_to_edges,
        hyperedges,
        candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
):
    """
    获取当前社区的扩展子图上下文，包含：
    1. 1-hop edges: 与当前社区直接相连的超边。
    2. Candidates: 1-hop edges 中的非社区节点。
    3. 2-hop edges: Candidates 参与的其他超边（不直接连社区）。
    获取局部结构后需要对其进行精简
    """
    current_community_set = set(current_community)

    # ==========================================
    # 1. 获取一跳超边 (1-hop edges)
    # ==========================================
    hop1_edge_indices = set()
    for node in current_community:
        if node in node_to_edges:
            hop1_edge_indices.update(node_to_edges[node])

    # ==========================================
    # 2. 识别候选节点 (Candidates)
    #    用于寻找二跳边
    # ==========================================
    candidates = set()
    for idx in hop1_edge_indices:
        edge_nodes = hyperedges[idx]
        for node in edge_nodes:
            if node not in current_community_set:
                candidates.add(node)
    print(f"候选节点:{candidates}")
    # ==========================================
    # 3. 获取二跳超边 (2-hop edges)
    # ==========================================
    hop2_edge_indices = set()
    for node in candidates:
        if node in node_to_edges:
            edge_indices = node_to_edges[node]
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
    hyperedges_1hop_dict = {idx: hyperedges[idx] for idx in hop1_edge_indices}
    hyperedges_2hop_dict = {idx: hyperedges[idx] for idx in hop2_edge_indices}
    local_hypergraph = list(hyperedges_1hop_dict.values()) + list(hyperedges_2hop_dict.values())
    # print(f"局部图结构（包含两跳超边）：{local_hypergraph}")

    # 由于只需要给出一跳超边，所以缩减时只缩减一跳超边
    # 这里所返回的是加入后社区模块度提升最大的前5个节点，如果这五个节点正好在第一阶段节点合并中有合并的节点，那么当其被选中时，需要同时将merge_records当中记录的结构相同的其他节点也一起加入到当前社区当中。
    # 这里有一个问题就是如果这里加入的有错误节点是否需要生成训练数据？暂定不管结构相同的节点加入是否正确。
    final_node, merge_records, mod_stats = utils.coarse_hypergraph_int_4_2hop_MC(
        neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, candidate_top_k, current_community)
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
        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
                edge_candidate_counts[edge_idx] += 1

    # 2. 统计每个节点的各项超边数据
    for node in final_node:
        hop1_count = 0
        hop2_count = 0
        shared_candidates_count = 0  # 新增：与其他候选节点共享的超边数目

        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
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

    return final_node, candidates_str, merge_records, mod_stats, candidates_stats


def get_community_state(current_community, node_to_edges, hyperedges):
    """计算社区的当前状态，包括规模和平均内部度"""
    size = len(current_community)
    if size == 0:
        return {"size": 0, "avg_internal_degree": 0.0}

    comm_set = set(current_community)
    int_degrees = []

    for node in comm_set:
        d_int = sum(1 for eid in node_to_edges[node] if all(n in comm_set for n in hyperedges[eid]))
        int_degrees.append(d_int)

    avg_internal_degree = float(np.mean(int_degrees))

    return {
        "size": size,
        "avg_internal_degree": round(avg_internal_degree, 3)
    }


def extract_nodes_from_output(output):
    """
    增强版提取函数：支持提取 Node_xxx，并能准确捕获 STOP 决策。
    返回: list[int] 或 ["STOP"]
    """
    if output is None:
        return []

    # 提取 Decision: 后的内容
    decision_match = re.search(r'Decision\s*:\s*(.*)', output, re.IGNORECASE | re.DOTALL)
    if not decision_match:
        return []

    decision_text = decision_match.group(1).upper()

    # 检查是否直接输出了 STOP
    if "STOP" in decision_text:
        return ["STOP"]

    # 提取数字节点
    matches = re.findall(r'NODE[_\s]*(\d+)', decision_text)

    seen = set()
    result = []
    for m in matches:
        n = int(m)
        if n not in seen:
            seen.add(n)
            result.append(n)

    return result


def save_data_entry(data_item, outfile):
    """使用 JSONL 格式追加写入，高效且安全"""
    """
        [高效版] 使用 JSONL 格式直接追加写入。
        不需要读取旧文件，直接在文件末尾加一行。
    """
    try:
        # 使用 'a' (append) 模式打开文件
        with open(outfile, 'a', encoding='utf-8') as f:
            # json.dumps 将字典转为字符串
            # 必须加上 + "\n" 来换行
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Save error: {e}")


def generate_dataset(
        seed_node,
        label,
        true_community,
        hyperedges,
        outfile1="contact_326_2.json",
        candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
):
    # 想要实现的功能：
    """
        生成社区扩张指令微调数据集：
        prompt：You are a community detection expert. Given the current community and its neighboring nodes with statistics, please choose the most likely node to be added next.
        调用API生成指令数据集
        输入给API大模型的内容：prompt+当前社区信息+候选节点信息+正确节点
        要求大模型根据正确答案输出分析过程，作为训练样本的COT链
    """
    instruction2 = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."
    # 1. 构建图索引,可以由节点获取超边id
    node_to_edges = build_hypergraph_index(hyperedges)
    # # 2. 模拟扩张步骤
    # # 真实场景中，我们不能把整个 ground truth community 都给模型。
    # # 从当前社区当中随机选择一个种子节点，从一个种子节点开始扩张
    initial_v0 = seed_node
    current_community = [initial_v0]
    print(f"当前社区节点: {current_community}")
    # 为了防止模型预测错误导致死循环，增加 max_steps 限制
    # 记录真实社区的集合，方便后续判断和计算
    true_community_set = set(true_community)
    current_top_k = candidate_top_k
    while True:
        # 获取当前上下文
        # 3. 获取当前社区的局部结构作为上下文信息，其中candidates_stats记录候选节点的信息{节点id：该节点所属社区},edge_list_str记录超边
        # 当前这一步返回了候选节点，候选节点状态和合并节点信息。这里需要补充候选节点信息，暂时不改也行
        candidates, candidates_state, merge_node, mod_stats, candidates_stats_dict = get_subgraph_context(
            current_community, node_to_edges, hyperedges, current_top_k)
        # 这里需要增加当前社区的状态
        comm_state = get_community_state(current_community, node_to_edges, hyperedges)
        # 这里记录候选节点的模块度整体变化属性
        comm_state["max_delta_m"] = mod_stats["max_delta_m"]
        comm_state["positive_ratio"] = mod_stats["positive_ratio"]
        if not candidates:
            print("No more neighbors found. Stopping expansion.")
            break
        input_data = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
            "true_community": str(true_community),
        }
        # 【核心修改 1】提前计算还剩下哪些真实节点没被找到
        remaining_global = list(true_community_set - set(current_community))
        # 判断是否应该 STOP
        is_stop = False
        target_nodes = []

        if len(remaining_global) == 0:
            # 【核心修改 2】不再判断长度，而是看剩余真实节点是否为 0
            # 真实社区确实已经全部找到了，理直气壮地 STOP
            is_stop = True
        else:
            # 【修改点 2】程序化自动挑选 1-4 个正确节点
            # 取真实社区集合与当前候选集集合的交集
            valid_candidates = list(true_community_set.intersection(candidates))

            if not valid_candidates:
                # 取真实社区集合与当前候选集集合的交集
                valid_candidates = list(true_community_set.intersection(candidates))

                if not valid_candidates:
                    # 直接处理无真实节点的情况：结构断层，执行强制架桥
                    print("⚠️ 候选集中无真实节点！判定为结构断层，执行强制架桥 (Global Fallback)。")
                    if remaining_global:
                        bridge_node = random.choice(remaining_global)
                        current_community.append(bridge_node)
                        print(f"➡ 强制架桥：拉取孤岛节点 Node_{bridge_node}")
                        continue
                    else:
                        is_stop = True
            else:
                # 核心规则挑选：优先选 内部连接(hop1)多、外部连接(hop2)少、且与其他候选节点共享多 的节点
                # 排序规则：按 hop1 降序，再按 shared_candidates 降序，最后按 hop2 升序
                valid_candidates.sort(
                    key=lambda x: (
                        candidates_stats_dict[x]['hop1'],
                        candidates_stats_dict[x]['shared_candidates'],
                        -candidates_stats_dict[x]['hop2']
                    ),
                    reverse=True
                )

                # 随机决定本次挑几个节点 (1-4 个，且不能超过有效节点总数)
                num_to_pick = min(random.randint(1, 4), len(valid_candidates))
                target_nodes = valid_candidates[:num_to_pick]

            # 【修改点 3】调用大模型生成 Reasoning (倒推生成)
        if is_stop:
            system_prompt = "You are an expert data annotator writing reasoning paths for AI training."
            user_prompt = f"""Given the following community state and candidate information:{json.dumps(input_data, indent=2)}
            Field definitions:
            - size: number of nodes in the current community
            - avg_internal_degree: average internal connectivity of the community
            - max_delta_m: maximum modularity gain from all candidates
            - positive_ratio: proportion of candidates with positive modularity contribution  
            Community Expansion Awareness (important):
            First infer the expansion stage of the community based on its structural state:
            - Early stage: small size and low avg_internal_degree → prioritize strong connectors
            - Late stage: high avg_internal_degree, low max_delta_m or low positive_ratio → only highly compatible nodes are beneficial        
            Ground truth: The correct action is to STOP expansion.
            Task:
            Provide a concise structural explanation (≤ 80 words) explaining why the community should STOP expanding.Focus on why further expansion would NOT improve structure.
            Strict constraints:
            - NEVER use phrases like "ground truth", "real community", "true labels"
            - Do NOT output any decision (e.g., STOP or node selection)
            - Do NOT list nodes
            - Output ONLY the reasoning text
            Output format:
            Reasoning:
            <your concise explanation>"""
            decision_str = "STOP"
        else:
            target_str = ",".join([f"Node_{n}" for n in target_nodes])
            system_prompt = "You are an expert data annotator writing reasoning paths for AI training."
            user_prompt = f"""Given the following community state and candidate information:{json.dumps(input_data, indent=2)}
            Field definitions:
            - size: number of nodes in the current community
            - avg_internal_degree: average internal connectivity of the community
            - max_delta_m: maximum modularity gain from all candidates
            - positive_ratio: proportion of candidates with positive modularity contribution
            Community Expansion Awareness (important):
            First infer the expansion stage of the community based on its structural state:
            - Early stage: small size and low avg_internal_degree → prioritize strong connectors
            - Late stage: high avg_internal_degree, low max_delta_m or low positive_ratio → only highly compatible nodes are beneficial
            Ground truth: The optimal nodes to add next are: {target_str}.
            Task:
            Provide a concise and structured explanation (≤ 80 words) detailing why these specific candidate nodes possess the best structural fit.
            Strict constraints:
            - NEVER use phrases like "ground truth", "real community", "true labels"
            - Do NOT output any decision (e.g., STOP or node selection)
            - Output ONLY the reasoning text
            Output format:
            Reasoning:
            <your concise explanation>"""
            decision_str = target_str

        try:
            # 让大模型只生成推理过程
            llm_reasoning = call_qwen_api(system_prompt, user_prompt)
            # 组合成最终我们希望微调模型学会的输出格式
            final_output = f" {llm_reasoning.strip()}\nDecision: {decision_str}"
        except Exception as e:
            print(f"API Call failed during reasoning generation: {e}")
            break

        # 保存这条完美的数据
        all_auto_added_nodes = []
        if not is_stop and merge_node:
            for pred_node in target_nodes:
                if pred_node in merge_node:
                    # 【核心修改 4】严格把关：只把同样属于真实社区的合并节点拉进来
                    # 避免噪声污染完美轨迹
                    for auto_node in merge_node[pred_node]:
                        if auto_node in true_community_set:
                            all_auto_added_nodes.append(auto_node)

        input_data2 = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
        }
        data_item = {
            "instruction": instruction2,
            "input": input_data2,
            "output": final_output,  # 完美的 推理 + 决策
            "meta": {
                "label": label,
                "added_node": target_nodes if not is_stop else ["STOP"],
                "auto_added_similar_nodes": all_auto_added_nodes,
                "is_stop_sample": is_stop
            }
        }
        save_data_entry(data_item, outfile1)
        print(f"✅ 保存样本 | 目标: {decision_str}")

        if is_stop:
            print("🏁 社区扩张完成 (STOP)。\n")
            break

        # 推进状态：将挑选的节点加入当前社区
        for node in target_nodes:
            if node not in current_community:
                current_community.append(node)

        for s_node in all_auto_added_nodes:
            if s_node not in current_community:
                current_community.append(s_node)

    return 1


def get_subgraph_context_with_exclusion(
        current_community,
        excluded_nodes,
        node_to_edges,
        hyperedges,
        candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
):
    """
    带剔除逻辑的子图上下文获取函数：
    确保 excluded_nodes（被剔除的真实节点）不会出现在候选集中，强迫模型面对纯噪声或边界。
    """
    current_community_set = set(current_community)
    excluded_nodes_set = set(excluded_nodes)

    # 1. 获取一跳超边
    hop1_edge_indices = set()
    for node in current_community:
        if node in node_to_edges:
            hop1_edge_indices.update(node_to_edges[node])

    # 2. 识别候选节点 (重点修改：拦截被剔除的节点)
    candidates = set()
    for idx in hop1_edge_indices:
        edge_nodes = hyperedges[idx]
        for node in edge_nodes:
            # 必须既不在当前社区，也不在我们故意剔除的名单中
            if node not in current_community_set and node not in excluded_nodes_set:
                candidates.add(node)

    # 3. 获取二跳超边
    hop2_edge_indices = set()
    for node in candidates:
        if node in node_to_edges:
            hop2_edge_indices.update(node_to_edges[node])
    hop2_edge_indices = hop2_edge_indices - hop1_edge_indices

    # 4. 采样与合并 (调用 utils)
    neighbors = list(candidates)
    hyperedges_1hop_dict = {idx: hyperedges[idx] for idx in hop1_edge_indices}
    hyperedges_2hop_dict = {idx: hyperedges[idx] for idx in hop2_edge_indices}

    # 注意：如果 candidates 为空，可以直接返回
    if not neighbors:
        return [], "", {}, {"max_delta_m": 0, "positive_ratio": 0}
    final_node, merge_records, mod_stats = utils.coarse_hypergraph_int_4_2hop_MC(
        neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, candidate_top_k, current_community)

    # 5. 统计与格式化
    candidates_stats = {}
    edge_candidate_counts = defaultdict(int)
    for node in final_node:
        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
                edge_candidate_counts[edge_idx] += 1

    for node in final_node:
        hop1_count = hop2_count = shared_candidates_count = 0
        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
                if edge_idx in hop1_edge_indices: hop1_count += 1
                if edge_idx in hop2_edge_indices: hop2_count += 1
                if edge_candidate_counts[edge_idx] > 1: shared_candidates_count += 1
        candidates_stats[node] = {"hop1": hop1_count, "hop2": hop2_count, "shared_candidates": shared_candidates_count}

    candidates_str_list = [
        f"Node {node}: Shared hyperedges with current community={s['hop1']}, External hyperedges={s['hop2']}, Shared hyperedges with other candidates={s['shared_candidates']}"
        for node, s in candidates_stats.items()
    ]
    return final_node, "\n".join(candidates_str_list), merge_records, mod_stats


def generate_batch_stop_samples(
        label,
        true_community,
        hyperedges,
        outfile="contact_stop_samples.json",
        candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
):
    """
    批量生成 STOP 样本的主逻辑：
    遍历完整社区，分别剔除 1 个和 2 个节点构造边界，测试大模型是否能果断选择 STOP。
    """
    instruction2 = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."
    node_to_edges = build_hypergraph_index(hyperedges)
    true_community_set = set(true_community)
    removal_cases = []

    # 策略 1：遍历剔除 1 个节点
    for n in true_community:
        removal_cases.append([n])

    # 策略 2：剔除 2 个节点 (为了防止API成本爆炸，这里对2节点组合进行随机采样，最多采样等同于社区大小的数量)
    if len(true_community) >= 2:
        all_pairs = list(itertools.combinations(true_community, 2))
        random.shuffle(all_pairs)
        removal_cases.extend(list(all_pairs[:4 * len(true_community)]))

    print(f"\n🚀 开始批量生成 STOP 样本 (社区 Label: {label}) | 待测边界陷阱数: {len(removal_cases)}")

    for excluded_nodes in removal_cases:
        # 构造当前社区（已挖去正确节点）
        current_community = list(true_community_set - set(excluded_nodes))
        # 获取上下文（传入 excluded_nodes 屏蔽名单）
        candidates, candidates_state, merge_node, mod_stats = get_subgraph_context_with_exclusion(
            current_community, excluded_nodes, node_to_edges, hyperedges, candidate_top_k
        )

        if not candidates:
            # 如果屏蔽后连候选节点都没了，就没有调大模型选择的意义了
            continue

        comm_state = get_community_state(current_community, node_to_edges, hyperedges)
        comm_state["max_delta_m"] = mod_stats["max_delta_m"]
        comm_state["positive_ratio"] = mod_stats["positive_ratio"]

        input_data = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
            # 注意：在生成数据时我们不要把 true_community 传给大模型，防止它依赖真实信息作弊
        }

        system_prompt = "You are an expert data annotator writing reasoning paths for AI training."
        user_prompt = f"""Given the following community state and candidate information:{json.dumps(input_data, indent=2)}
                    Field definitions:
                    - size: number of nodes in the current community
                    - avg_internal_degree: average internal connectivity of the community
                    - max_delta_m: maximum modularity gain from all candidates
                    - positive_ratio: proportion of candidates with positive modularity contribution  
                    Community Expansion Awareness (important):
                    First infer the expansion stage of the community based on its structural state:
                    - Early stage: small size and low avg_internal_degree → prioritize strong connectors
                    - Late stage: high avg_internal_degree, low max_delta_m or low positive_ratio → only highly compatible nodes are beneficial        
                    Ground truth: The correct action is to STOP expansion.
                    Task:
                    Provide a concise structural explanation (≤ 80 words) explaining why the community should STOP expanding.Focus on why further expansion would NOT improve structure.
                    Strict constraints:
                    - NEVER use phrases like "ground truth", "real community", "true labels"
                    - Do NOT output any decision (e.g., STOP or node selection)
                    - Do NOT list nodes
                    - Output ONLY the reasoning text
                    Output format:
                    Reasoning:
                    <your concise explanation>"""
        decision_str = "STOP"

        is_stop = True
        try:
            # 让大模型只生成推理过程
            llm_reasoning = call_qwen_api(system_prompt, user_prompt)
            # 组合成最终我们希望微调模型学会的输出格式
            final_output = f" {llm_reasoning.strip()}\nDecision: {decision_str}"
        except Exception as e:
            print(f"API Call failed during reasoning generation: {e}")
            break

        input_data2 = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
        }
        data_item = {
            "instruction": instruction2,
            "input": input_data2,
            "output": final_output,  # 完美的 推理 + 决策
            "meta": {
                "label": label,
                "added_node": ["STOP"],
                "auto_added_similar_nodes": [],
                "is_stop_sample": is_stop
            }
        }
        save_data_entry(data_item, outfile)


def main(argv=None):
    args = parse_args(argv)
    # 读取文件路径
    base_path = args.base_path
    label_path = os.path.join(base_path, "labels.txt")
    hyperedge_path = os.path.join(base_path, "hyperedges.txt")
    output_dir = args.output_dir
    # 如果目录不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)
    outfile1 = os.path.join(output_dir, args.data_output)
    outfile2 = os.path.join(output_dir, args.stop_output)

    # normpath 会去掉路径末尾多余的斜杠，basename 获取最后一级文件夹名
    dataname = os.path.basename(os.path.normpath(base_path))
    community_file_path = os.path.join(output_dir, f"{dataname}_community_split.json")
    # 读取节点标签，处理成节点id为键，对应标签为值
    label = defaultdict()
    with open(label_path, 'r', encoding='utf-8') as file:
        # 每行读到的数放在line当中
        for line_num, line in enumerate(file, start=1):  # 从1开始计数
            line = line.strip()
            if line:  # 如果行不为空
                label[line_num] = line  # 行号作为键

    # 读取超边集,以二维列表方式存储
    hyperedges = []
    with open(hyperedge_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除首尾空白字符和换行符
            if line:  # 跳过空行
                # 分割字符串并转换为整数
                row = [int(x) for x in line.split(',')]
                hyperedges.append(row)

    # print(hyperedges)

    # 将节点按照标签分组,即将键按照值分组
    label_to_nodes = {}
    for node, label1 in label.items():
        if label1 not in label_to_nodes:
            label_to_nodes[label1] = []
        label_to_nodes[label1].append(node)

    # 检查文件是否存在
    if os.path.exists(community_file_path):
        print(f"检测到已存在的社区划分文件 '{community_file_path}'，正在读入...")

        with open(community_file_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        train_labels = split_data["train"]
        test_labels = split_data["test"]

        # 根据标签恢复出 train_communities 和 test_communities
        train_communities = OrderedDict((k, label_to_nodes[k]) for k in train_labels if k in label_to_nodes)
        test_communities = OrderedDict((k, label_to_nodes[k]) for k in test_labels if k in label_to_nodes)

        print(f"读入完成。Train communities: {len(train_communities)}, Test communities: {len(test_communities)}")

    else:
        print(f"未检测到历史划分文件，开始进行首次社区划分与过滤...")

        # 1. 过滤：只保留节点数量在 10 到 300 之间的社区
        true_community = OrderedDict(
            (k, v) for k, v in sorted(label_to_nodes.items())
            if 10 <= len(v) <= 300
        )

        # 2. 打乱，固定随机种子保证可复现
        random.seed(42)
        community_items = list(true_community.items())
        random.shuffle(community_items)

        # 3. 按比例（6:4）划分
        train_ratio = 0.6
        split_idx = int(len(community_items) * train_ratio)
        train_items = community_items[:split_idx]
        test_items = community_items[split_idx:]

        # 4. 转回 OrderedDict
        train_communities = OrderedDict(train_items)
        test_communities = OrderedDict(test_items)

        # 5. 持久化到统一的输出路径
        with open(community_file_path, "w", encoding="utf-8") as f:
            json.dump({
                "train": list(train_communities.keys()),
                "test": list(test_communities.keys())
            }, f, indent=2, ensure_ascii=False)

        print(f"首次划分并保存至 '{community_file_path}' 成功！")
        print(f"Total filtered communities: {len(true_community)}")
        print(f"Train communities: {len(train_communities)}")
        print(f"Test communities: {len(test_communities)}")

    # 如果外部 args 传入了指定的 train_labels，则覆盖并过滤；否则默认使用上面划分/读取好的社区
    # 如果用户没有在命令行输入 --train_labels，parse_label_list(args.train_labels) 会返回空集合
    specified_labels = parse_label_list(args.train_labels)

    if specified_labels:
        # 只有当用户真的显式指定了标签时（集合不为空），才去覆盖和过滤
        train_communities = {
            label: nodes
            for label, nodes in label_to_nodes.items()
            if str(label) in specified_labels
        }
        print(f"应用了外部指定的训练标签，当前执行社区数量: {len(train_communities)}")
    else:
        # 如果集合为空，则静默使用前面通过 json 读入或首次自动划分好的 train_communities
        print(f"未指定外部标签，将采用系统自动划分/读取的 {len(train_communities)} 个训练社区进行后续计算")

    # ==================== 4. 数据生成逻辑 ====================
    for label2, nodes in train_communities.items():
        label_str = str(label2)
        for seed_node in nodes:
            # 【注意】请确保 generate_dataset 内部写入文件时使用的是追加模式 ('a')
            generate_dataset(
                seed_node,
                label2,
                nodes,
                hyperedges,
                outfile1,
                candidate_top_k=args.candidate_top_k,
            )

        generate_batch_stop_samples(
            label2,
            nodes,
            hyperedges,
            outfile2,
            candidate_top_k=args.candidate_top_k,
        )


if __name__ == "__main__":
    main()

