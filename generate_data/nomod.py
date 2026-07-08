import sys
import requests
import os.path
from collections import defaultdict, OrderedDict
import random
import json
import copy
import random
from collections import defaultdict
from typing import Tuple, List, Any, Dict
import re
import numpy as np
import itertools

sys.stdout.reconfigure(encoding='utf-8')


def call_qwen_api(system_prompt, user_prompt):
    """调用通义千问 (Qwen) API"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {'sk-118f497d49394c0bad1b684d0433a950'}"
    }

    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.8
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120, stream=True)
        res_json = response.json()

        if 'choices' not in res_json:
            error_msg = res_json.get('error', {}).get('message', '未知错误')
            print(f"⚠️ API 返回异常: {error_msg}")
            return None

        return res_json['choices'][0]['message']['content']

    except Exception as e:
        print(f"❌ 网络层错误: {e}")
        return None


def build_hypergraph_index(hyperedges):
    """构建倒排索引：节点 -> 所在的超边ID列表"""
    node_to_edges = defaultdict(list)
    for edge_idx, nodes in enumerate(hyperedges):
        for node in nodes:
            node_to_edges[node].append(edge_idx)
    return node_to_edges


def get_subgraph_context(current_community, node_to_edges, hyperedges, top_k=6):
    """
    获取当前社区的扩展子图上下文
    【核心修改】保留 utils 的节点合并逻辑，但将筛选方式改为随机筛选
    """
    current_community_set = set(current_community)

    # 1. 获取一跳超边
    hop1_edge_indices = set()
    for node in current_community:
        if node in node_to_edges:
            hop1_edge_indices.update(node_to_edges[node])

    # 2. 识别候选节点
    candidates = set()
    for idx in hop1_edge_indices:
        edge_nodes = hyperedges[idx]
        for node in edge_nodes:
            if node not in current_community_set:
                candidates.add(node)
    print(f"候选节点:{candidates}")

    # 3. 获取二跳超边
    hop2_edge_indices = set()
    for node in candidates:
        if node in node_to_edges:
            edge_indices = node_to_edges[node]
            hop2_edge_indices.update(edge_indices)

    hop2_edge_indices = hop2_edge_indices - hop1_edge_indices

    neighbors = list(candidates)

    # 准备字典格式
    hyperedges_1hop_dict = {idx: hyperedges[idx] for idx in hop1_edge_indices}
    hyperedges_2hop_dict = {idx: hyperedges[idx] for idx in hop2_edge_indices}

    # 【修改点】传入 len(neighbors) 作为数量限制，迫使 utils 返回经过合并后的“所有”候选节点
    all_coarse_nodes, merge_records, mod_stats = coarse_hypergraph_int_4_2hop_MC(
        neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, len(neighbors), current_community
    )

    # 【修改点】从合并后的所有超级节点中，随机安全地过滤并抽出 top_k 个
    sample_size = min(top_k, len(all_coarse_nodes))
    final_node = random.sample(all_coarse_nodes, sample_size) if sample_size > 0 else []

    # 4. 统计随机筛选后的每个节点的各项超边数据
    candidates_stats = {}
    edge_candidate_counts = defaultdict(int)
    for node in final_node:
        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
                edge_candidate_counts[edge_idx] += 1

    for node in final_node:
        hop1_count = 0
        hop2_count = 0
        shared_candidates_count = 0

        if node in node_to_edges:
            for edge_idx in node_to_edges[node]:
                if edge_idx in hop1_edge_indices:
                    hop1_count += 1
                if edge_idx in hop2_edge_indices:
                    hop2_count += 1
                if edge_candidate_counts[edge_idx] > 1:
                    shared_candidates_count += 1

        candidates_stats[node] = {
            "hop1": hop1_count,
            "hop2": hop2_count,
            "shared_candidates": shared_candidates_count
        }

    # 5. 格式化 Candidates 字符串
    candidates_str_list = []
    for node, stats in candidates_stats.items():
        hop1_count = stats["hop1"]
        hop2_count = stats["hop2"]
        shared_cands = stats["shared_candidates"]

        candidates_str_list.append(
            f"Node {node}: "
            f"Shared hyperedges with current community={hop1_count}, "
            f"External hyperedges={hop2_count}, "
            f"Shared hyperedges with other candidates={shared_cands}"
        )

    candidates_str = "\n".join(candidates_str_list)
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
    """增强版提取函数：支持提取 Node_xxx，并能准确捕获 STOP 决策。"""
    if output is None:
        return []

    decision_match = re.search(r'Decision\s*:\s*(.*)', output, re.IGNORECASE | re.DOTALL)
    if not decision_match:
        return []

    decision_text = decision_match.group(1).upper()
    if "STOP" in decision_text:
        return ["STOP"]

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
    """使用 JSONL 格式追加写入"""
    try:
        with open(outfile, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Save error: {e}")

def coarse_hypergraph_int_4_2hop_MC(
        node_1hop: list[int],
        hyperedges_1hop_dict: dict[int, list[int]],
        hyperedges_2hop_dict: dict[int, list[int]],
        K: int = 6,
        current_community: list[int] | None = None,
) -> tuple[list[int], dict[Any, list], dict[str, float]]:
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


def generate_dataset(seed_node, label, true_community, hyperedges, outfile1="contact_326_2.json"):
    # 想要实现的功能：
    """
        生成社区扩张指令微调数据集：
        prompt：You are a community detection expert. Given the current community and its neighboring nodes with statistics, please choose the most likely node to be added next.
        调用API生成指令数据集
        输入给API大模型的内容：prompt+当前社区信息+候选节点信息+正确节点
        要求大模型根据正确答案输出分析过程，作为训练样本的COT链
    """
    instruction2 = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."
    # 获取社区大小，要将社区中所有节点扩张
    num_com = len(true_community)
    # 1. 构建图索引,可以由节点获取超边id
    node_to_edges = build_hypergraph_index(hyperedges)
    # # 2. 模拟扩张步骤
    # # 从当前社区当中随机选择一个种子节点，从一个种子节点开始扩张
    initial_v0 = seed_node
    current_community = [initial_v0]
    print(f"当前社区节点: {current_community}")

    # 记录真实社区的集合，方便后续判断和计算
    true_community_set = set(true_community)
    # 固定候选集数目为 6
    base_top_k = 6
    current_top_k = base_top_k

    # 新增：重试计数器及上限
    MAX_RETRY = 10  # 最大连续重试次数
    retry_count = 0  # 当前连续重试次数

    while True:
        # 获取当前上下文 (内部包含随机选择 K=6 的逻辑)
        candidates, candidates_state, merge_node, mod_stats, candidates_stats_dict = get_subgraph_context(
            current_community, node_to_edges, hyperedges, current_top_k
        )

        # 获取当前社区的状态
        comm_state = get_community_state(current_community, node_to_edges, hyperedges)
        # 记录候选节点的模块度整体变化属性
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

        # 提前计算还剩下哪些真实节点没被找到
        remaining_global = list(true_community_set - set(current_community))
        is_stop = False
        target_nodes = []

        if len(remaining_global) == 0:
            is_stop = True
        else:
            # 取当前抽样出的随机候选集与真实社区的交集
            valid_candidates = list(true_community_set.intersection(candidates))

            if not valid_candidates:
                # 检查整个一跳邻域里是否还有真实节点
                hop1_edge_indices = set()
                for node in current_community:
                    if node in node_to_edges:
                        hop1_edge_indices.update(node_to_edges[node])

                all_neighborhood_candidates = set()
                for idx in hop1_edge_indices:
                    for node in hyperedges[idx]:
                        if node not in current_community:
                            all_neighborhood_candidates.add(node)

                actual_true_in_neighborhood = true_community_set.intersection(all_neighborhood_candidates)

                if actual_true_in_neighborhood:
                    retry_count += 1
                    if retry_count > MAX_RETRY:
                        # 超过重试上限，强制从邻域真实节点中桥接一个
                        bridge_node = random.choice(list(actual_true_in_neighborhood))
                        print(f"🔁 重试 {MAX_RETRY} 次未抽中，强制架桥节点 {bridge_node}")
                        current_community.append(bridge_node)
                        retry_count = 0
                        # ✅ 必须 continue，跳过本轮样本生成
                        continue
                    else:
                        print(f"🎲 随机筛选未抽中真实节点（邻域内存在），第 {retry_count}/{MAX_RETRY} 次重试...")
                        # ✅ 必须 continue，跳过本轮样本生成
                        continue
                else:
                    # 整个邻域都没有真实节点，全局强制架桥
                    print("⚠️ 整个邻域内已无真实节点！判定为结构断层，执行强制架桥 (Global Fallback)。")
                    if remaining_global:
                        bridge_node = random.choice(remaining_global)
                        current_community.append(bridge_node)
                        # ✅ 必须 continue，跳过本轮样本生成
                        continue
                    else:
                        is_stop = True
            else:
                # 成功抽中真实节点，重置重试计数器
                retry_count = 0

                # 原有的精细排序逻辑
                valid_candidates.sort(
                    key=lambda x: (
                        candidates_stats_dict[x]['hop1'],
                        candidates_stats_dict[x]['shared_candidates'],
                        -candidates_stats_dict[x]['hop2']
                    ),
                    reverse=True
                )
                num_to_pick = min(random.randint(1, 4), len(valid_candidates))
                target_nodes = valid_candidates[:num_to_pick]

        # 后续调用大模型、拼接 Reasoning 和状态推进逻辑保持完全一致
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
            user_prompt = f"""
            Given the following community state and candidate information:{json.dumps(input_data, indent=2)}
            Field definitions:
            - size: number of nodes in the current community
            - avg_internal_degree: average internal connectivity of the community
            - max_delta_m: maximum modularity gain from all candidates
            - positive_ratio: proportion of candidates with positive modularity contribution
            Community Expansion Awareness (important):
            First infer the expansion stage of the community based on its structural state:
            - Early stage: small size and low avg_internal_degree → prioritize strong connectors.
            - Late stage: high avg_internal_degree, low max_delta_m or low positive_ratio → only highly compatible nodes are beneficial.
            Ground truth: The optimal nodes to add next are: {target_str}.
            Task:
            Provide a concise, structured explanation (≤ 80 words). Evaluate the $avg\_internal\_degree$ improvement rate; candidates must intensify internal density relative to community expansion. If a node increases size while diluting connectivity, it constitutes a noisy expansion. Contrast 'shared edges with community' against 'shared edges with other candidates.' Preclude clusters that exhibit high internal cohesion but weak coupling to the current community to prevent drift. Finally, neutralize 'Hub' nodes with excessive external connectivity to avoid false bridges.
            Strict constraints:
            - NEVER use phrases like "ground truth", "real community", "true labels"
            - Do NOT output any decision (e.g., STOP or node selection)
            - Output ONLY the reasoning text
            Output format:
            Reasoning:
            <your concise explanation>
            """
            decision_str = target_str

        try:
            llm_reasoning = call_qwen_api(system_prompt, user_prompt)
            final_output = f" {llm_reasoning.strip()}\nDecision: {decision_str}"
        except Exception as e:
            print(f"API Call failed during reasoning generation: {e}")
            break

        all_auto_added_nodes = []
        if not is_stop and merge_node:
            for pred_node in target_nodes:
                if pred_node in merge_node:
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
            "output": final_output,
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

        for node in target_nodes:
            if node not in current_community:
                current_community.append(node)

        for s_node in all_auto_added_nodes:
            if s_node not in current_community:
                current_community.append(s_node)

    return 1


def get_subgraph_context_with_exclusion(current_community, excluded_nodes, node_to_edges, hyperedges):
    """
    带剔除逻辑的子图上下文获取函数
    【核心修改】同样调整为先利用 utils 完整合并，再进行随机筛选
    """
    current_community_set = set(current_community)
    excluded_nodes_set = set(excluded_nodes)

    hop1_edge_indices = set()
    for node in current_community:
        if node in node_to_edges:
            hop1_edge_indices.update(node_to_edges[node])

    candidates = set()
    for idx in hop1_edge_indices:
        edge_nodes = hyperedges[idx]
        for node in edge_nodes:
            if node not in current_community_set and node not in excluded_nodes_set:
                candidates.add(node)

    hop2_edge_indices = set()
    for node in candidates:
        if node in node_to_edges:
            hop2_edge_indices.update(node_to_edges[node])
    hop2_edge_indices = hop2_edge_indices - hop1_edge_indices

    neighbors = list(candidates)
    if not neighbors:
        return [], "", {}, {"max_delta_m": 0, "positive_ratio": 0}

    hyperedges_1hop_dict = {idx: hyperedges[idx] for idx in hop1_edge_indices}
    hyperedges_2hop_dict = {idx: hyperedges[idx] for idx in hop2_edge_indices}

    # 解除截断，完整获取合并后的全部超级节点
    top_K = 6
    all_coarse_nodes, merge_records, mod_stats = coarse_hypergraph_int_4_2hop_MC(
        neighbors, hyperedges_1hop_dict, hyperedges_2hop_dict, len(neighbors), current_community
    )

    # 从中随机筛选出指定数量的候选节点
    sample_size = min(top_K, len(all_coarse_nodes))
    final_node = random.sample(all_coarse_nodes, sample_size) if sample_size > 0 else []

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


def generate_batch_stop_samples(label, true_community, hyperedges, outfile="contact_stop_samples.json"):
    instruction2 = "You are a Hypergraph Community Detection Expert. Given the current community nodes, candidate neighbors, and the local hypergraph structure, your task is to select 1-4 nodes to expand the community. Only choose from the listed candidates or STOP."
    node_to_edges = build_hypergraph_index(hyperedges)
    true_community_set = set(true_community)
    removal_cases = []

    for n in true_community:
        removal_cases.append([n])

    if len(true_community) >= 2:
        all_pairs = list(itertools.combinations(true_community, 2))
        random.shuffle(all_pairs)
        removal_cases.extend(list(all_pairs[:4 * len(true_community)]))

    print(f"\n🚀 开始批量生成 STOP 样本 (社区 Label: {label}) | 待测边界陷阱数: {len(removal_cases)}")

    for excluded_nodes in removal_cases:
        current_community = list(true_community_set - set(excluded_nodes))
        candidates, candidates_state, merge_node, mod_stats = get_subgraph_context_with_exclusion(
            current_community, excluded_nodes, node_to_edges, hyperedges
        )

        if not candidates:
            continue

        comm_state = get_community_state(current_community, node_to_edges, hyperedges)
        comm_state["max_delta_m"] = mod_stats["max_delta_m"]
        comm_state["positive_ratio"] = mod_stats["positive_ratio"]

        input_data = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
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
            llm_reasoning = call_qwen_api(system_prompt, user_prompt)
            final_output = f" {llm_reasoning.strip()}\nDecision: {decision_str}"
        except Exception as e:
            print(f"API Call failed during reasoning generation: {e}")
            continue

        input_data2 = {
            "current_community": str(current_community),
            "community_state": comm_state,
            "candidates": candidates_state,
        }
        data_item = {
            "instruction": instruction2,
            "input": input_data2,
            "output": final_output,
            "meta": {
                "label": label,
                "added_node": ["STOP"],
                "auto_added_similar_nodes": [],
                "is_stop_sample": is_stop
            }
        }
        save_data_entry(data_item, outfile)


def main():
    base_path = r"D:\InstructCom\datasets\synthetic-2000-20"
    label_path = os.path.join(base_path, "2000-20-node-labels.txt")
    hyperedge_path = os.path.join(base_path, "2000-20-hyperedges.txt")
    output_dir = "D:\\InstructCom\\src\\2000-20"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outfile1 = os.path.join(output_dir, "612-nomod.json")
    outfile2 = os.path.join(output_dir, "stop-612-nomod.json")

    dataname = os.path.basename(os.path.normpath(base_path))
    label = defaultdict()
    with open(label_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if line:
                label[line_num] = line

    hyperedges = []
    with open(hyperedge_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                row = [int(x) for x in line.split(',')]
                hyperedges.append(row)

    label_to_nodes = {}
    for node, label1 in label.items():
        if label1 not in label_to_nodes:
            label_to_nodes[label1] = []
        label_to_nodes[label1].append(node)

    # true_community = OrderedDict(
    #     (k, v) for k, v in sorted(label_to_nodes.items())
    #     if 10 <= len(v) <= 100
    # )

    with open(r"D:\InstructCom\src\2000-20\synthetic-dataset_community_split.json", "r",
              encoding="utf-8") as f:
        split_data = json.load(f)
    train_labels = list(dict.fromkeys(split_data["train"]))
    # train_labels = {"6"}
    train_communities = {
        label: nodes
        for label, nodes in label_to_nodes.items()
        if str(label) in train_labels
    }

    for label2, nodes in train_communities.items():
        sampled_nodes = random.sample(nodes, 20)
        for seed_node in sampled_nodes:
            generate_dataset(seed_node, label2, nodes, hyperedges, outfile1)
        generate_batch_stop_samples(label2, nodes, hyperedges, outfile2)


if __name__ == "__main__":
    main()