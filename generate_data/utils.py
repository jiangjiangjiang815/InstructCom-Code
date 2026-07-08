# 缩减算法，节点合并和候选节点筛选
import copy
import random
from collections import defaultdict
from typing import Tuple, List, Any, Dict

import numpy as np


def coarse_hypergraph_int_4_2hop_MC(
        node_1hop: list[int],
        hyperedges_1hop_dict: dict[int, list[int]],
        hyperedges_2hop_dict: dict[int, list[int]],
        K:int=6,
        current_community: list[int] | None = None,
) -> tuple[list[int], list[Any], dict[str, float]] | tuple[
    list[int], defaultdict[Any, list], dict[str, float] | dict[str, float | Any]]:
    # 增加模块度筛选，选择提升社区模块度最大的前K个节点
    # 1. 创建本地副本
    local_hyperedges_1hop = copy.deepcopy(hyperedges_1hop_dict)
    local_hyperedges_2hop = copy.deepcopy(hyperedges_2hop_dict)
    node_1hop = sorted(node_1hop)
    # 第一阶段还是进行结构相似节点的合并
    # 2. 计算节点度（固定顺序遍历）
    degree_dict = {node: 0 for node in node_1hop}

    # 模拟 Julia 的 sort(collect(dict)) 行为，按 Key 排序遍历
    for _, hyperedge in sorted(local_hyperedges_1hop.items()):
        for node in hyperedge:
            if node in degree_dict:
                degree_dict[node] += 1

    # 按度分组节点
    group_dict = defaultdict(list)
    for node, degree in sorted(degree_dict.items()):
        group_dict[degree].append(node)

    # 3. 计算超边权重
    hyperedge_weights = {}
    # 遍历 2-hop 超边
    # 1. 弃用 labels，直接使用传入的 current_community
    # combined_hyperedges_dict = {**hyperedges_1hop_dict, **hyperedges_2hop_dict}
    community_nodes = set(current_community) if current_community else set()

    hyperedge_weights = {}
    for hyper_id, hyperedge in hyperedges_1hop_dict.items():
        hyperedge_count = len(hyperedge)
        # 计算超边中有多少节点属于【当前维护的社区 C】，而不是标签
        community_node_count = sum(1 for node in hyperedge if node in community_nodes)

        if community_node_count > 0:
            weight = community_node_count / hyperedge_count
        else:
            weight = -1.0
        hyperedge_weights[hyper_id] = weight

    # 计算超边尺寸 (degree of hyperedge)
    hyperedge_degree_map = {hid: len(he) for hid, he in hyperedges_1hop_dict.items()}

    # 构建节点到超边的映射 (只针对 node_1hop 中的节点)
    node_to_hyperedges = defaultdict(list)
    for hyper_id, hyperedge in hyperedges_1hop_dict.items():
        for node in hyperedge:
            if node in degree_dict:  # degree_dict keys correspond to node_1hop
                node_to_hyperedges[node].append(hyper_id)

    # 4. 计算节点相似度
    all_similarities = []

    # 按度数分组遍历
    for degree in sorted(group_dict.keys()):
        nodes_in_group = group_dict[degree]
        n_group = len(nodes_in_group)

        for i in range(n_group):
            for j in range(i + 1, n_group):
                first_node = nodes_in_group[i]
                second_node = nodes_in_group[j]

                # 构建特征向量
                vec1_dict = defaultdict(list)
                vec2_dict = defaultdict(list)

                # 填充 vec1_dict
                for hid in node_to_hyperedges.get(first_node, []):
                    hdeg = hyperedge_degree_map[hid]
                    vec1_dict[hdeg].append(hyperedge_weights[hid])

                # 填充 vec2_dict
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

                    # 模拟 Julia 的 resize! (填充 0.0 或默认值) 并 sort!
                    # 注意：Julia resize! 增加长度时通常是不确定值，但在特征对齐上下文中，通常意味着填充空缺。
                    # 这里我们用 -2.0 填充(比 -1.0 小) 或者 0.0，这里采用 0.0 以保持对其
                    # 但考虑到 weight 可能为 -1.0，用一个极小值填充可能更安全，
                    # 不过参照常规图算法，这里假设补齐特征为 0.0 (无权重)
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

                # 计算相似度
                similarity = 0.0
                if len(vec1) == 1 and len(vec2) == 1:
                    # 欧式距离相似度
                    dist = np.linalg.norm(vec1 - vec2)
                    similarity = 1.0 / (1.0 + dist)
                elif len(vec1) > 0 and len(vec2) > 0:
                    # 余弦相似度
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
    # protect_set = set(protect_node)

    for sim in all_similarities:
        if sim['similarity'] < SIMILARITY_THRESHOLD:
            continue

        node1, node2 = sim['node_pair']

        if node1 in removed_nodes or node2 in removed_nodes:
            continue

        # 随机选择保留的节点
        keep_idx = random.choice([0, 1])
        keep_node = sim['node_pair'][keep_idx]
        remove_node = sim['node_pair'][1 - keep_idx]

        merge_records[keep_node].append(remove_node)
        merged_nodes_map[remove_node] = keep_node
        removed_nodes.add(remove_node)

        # 从 1-hop 超边中移除节点
        for hyperedge in local_hyperedges_1hop.values():
            if remove_node in hyperedge:
                # 注意：要在列表上进行移除，建议重建列表或小心使用 remove
                hyperedge[:] = [x for x in hyperedge if x != remove_node]

        # 从 2-hop 超边中移除节点
        for hyperedge in local_hyperedges_2hop.values():
            if remove_node in hyperedge:
                hyperedge[:] = [x for x in hyperedge if x != remove_node]

    # 6. 移除孤立节点
    # 检查剩余的 node_1hop 节点是否还连接在任意 1-hop 超边上
    for node in node_1hop:
        if node not in removed_nodes:
            connected = False
            for hyperedge in local_hyperedges_1hop.values():
                if node in hyperedge:
                    connected = True
                    break

            if not connected:
                removed_nodes.add(node)
                # 从所有超边中清理（保险起见）
                for hyperedge in local_hyperedges_1hop.values():
                    if node in hyperedge:
                        hyperedge[:] = [x for x in hyperedge if x != node]
                for hyperedge in local_hyperedges_2hop.values():
                    if node in hyperedge:
                        hyperedge[:] = [x for x in hyperedge if x != node]

    # 输出合并结果统计
    new_nodes = sorted(list(set(node_1hop) - removed_nodes))
    print(f"\n合并结果：共保留节点数 = {len(new_nodes)}，被合并节点数 = {len(removed_nodes)}")

    # new_nodes作为已经经过一轮缩减的候选节点，再继续从中筛选出模块度提升最大的前K个节点。
    # 如果没有提供当前社区节点，无法进行模块度评估，直接按度数保留 (Fallback)
    if current_community is None or len(current_community) == 0:
        return new_nodes[:K], [], {"max_delta_m": 0.0, "positive_ratio": 0.0}

    # 2. 识别当前目标社区节点 C (直接转为 set，O(1) 查找，极其高效)
    community_nodes = set(current_community)

    # 合并 1-hop 和 2-hop 超边用于全局评估
    all_hyperedges = {**local_hyperedges_1hop, **local_hyperedges_2hop}

    # 构建 节点 -> 超边映射 以加速计算 O(N)
    node_to_he = defaultdict(list)
    # 预计算每条超边中属于社区 C 的节点数量
    he_intersect_counts = {}

    e_in_base = 0.0
    e_out_base = 0.0

    # 3. 初始化基础超图局部模块度 M_C
    for hid, he in all_hyperedges.items():
        he_len = len(he)
        if he_len == 0:
            continue

        for n in he:
            node_to_he[n].append(hid)

        count_c = sum(1 for n in he if n in community_nodes)
        he_intersect_counts[hid] = count_c

        # 超图中的 e_in 和 e_out 定义：以所占节点比例作为权重
        if count_c > 0:
            e_in_base += count_c / he_len
            e_out_base += (he_len - count_c) / he_len

    base_hmc = e_in_base / e_out_base if e_out_base > 0 else 0.0

    # 4. 计算每个候选节点的 模块度增益 ΔM
    node_scores = []

    for v in new_nodes:
        new_e_in = e_in_base
        new_e_out = e_out_base

        if v not in community_nodes:
            # 评估节点 v 加入社区带来的增益: M_{C U {v}} - M_C
            for hid in node_to_he.get(v, []):
                he_len = len(all_hyperedges[hid])
                count_c = he_intersect_counts[hid]

                if count_c > 0:
                    new_e_in += 1.0 / he_len
                    new_e_out -= 1.0 / he_len
                else:
                    # 之前超边完全在外部，现在和 C 产生了交集
                    new_e_in += 1.0 / he_len
                    new_e_out += (he_len - 1.0) / he_len

            new_hmc = new_e_in / new_e_out if new_e_out > 0 else float('inf')
            delta_m = new_hmc - base_hmc

        else:
            # 如果节点 v 已经在社区中，评估它存在的贡献: M_C - M_{C \ {v}}
            new_e_in_rem = e_in_base
            new_e_out_rem = e_out_base
            for hid in node_to_he.get(v, []):
                he_len = len(all_hyperedges[hid])
                count_c = he_intersect_counts[hid]

                if count_c > 1:
                    new_e_in_rem -= 1.0 / he_len
                    new_e_out_rem += 1.0 / he_len
                elif count_c == 1:
                    # 移除后，该超边彻底失去与 C 的交集
                    new_e_in_rem -= 1.0 / he_len
                    new_e_out_rem -= (he_len - 1.0) / he_len

            hmc_rem = new_e_in_rem / new_e_out_rem if new_e_out_rem > 0 else 0.0
            delta_m = base_hmc - hmc_rem

        node_scores.append({
            'node': v,
            'delta_m': delta_m
        })

    # ==========================================
    # [新增核心修改] 5. 计算社区停止标志统计信息
    # ==========================================
    if not node_scores:
        mod_stats = {"max_delta_m": 0.0, "positive_ratio": 0.0}
    else:
        # 提取所有节点的模块度增量
        all_deltas = [record['delta_m'] for record in node_scores]

        # 最大的模块度增量
        max_delta_m = max(all_deltas)

        # 正向增益节点的比例
        positive_count = sum(1 for d in all_deltas if d > 0)
        positive_ratio = positive_count / len(all_deltas)

        mod_stats = {
            "max_delta_m": round(max_delta_m, 4),
            "positive_ratio": round(positive_ratio, 3)
        }

    # 5. 根据 ComGPT 思想筛选 Potential Nodes (ΔM > 0)
    # 只选择模块度大于0的节点会出现候选节点过于少的情况
    potential_nodes = [record for record in node_scores if record['delta_m'] > 0]

    # 兜底：如果都没有正增益，退化为选负增益最小的
    if not potential_nodes:
        potential_nodes = node_scores

    # 对所有候选节点进行选择，选出 Top-K 节点
    node_scores.sort(key=lambda x: x['delta_m'], reverse=True)
    keep_nodes = [record['node'] for record in node_scores[:K]]
    print(keep_nodes)

    keep_nodes_set = set(keep_nodes)
    removed_nodes = set(new_nodes) - keep_nodes_set

    # 6. 从超边中移除落选的节点
    for hyperedge in local_hyperedges_1hop.values():
        hyperedge[:] = [x for x in hyperedge if x not in removed_nodes]
    for hyperedge in local_hyperedges_2hop.values():
        hyperedge[:] = [x for x in hyperedge if x not in removed_nodes]

    # 输出合并结果统计
    print(f"\n模块度筛选结果：共保留节点数 = {len(keep_nodes)}，因模块度增益不足被移除 = {len(removed_nodes)}")

    final_nodes = sorted(keep_nodes)

    # 7. 处理和过滤超边
    test_hyperedges = [sorted(he) for he in local_hyperedges_1hop.values() if len(he) > 0]
    print(f"\n合并前一跳超边数量= {len(local_hyperedges_1hop)}, 经过节点合并后的一跳超边数量= {len(test_hyperedges)}")

    # 还需要保留二跳超边
    final_hyperedges = []
    for he in local_hyperedges_1hop.values():
        # 过滤已被移除的节点 (其实上面已经 inplace 修改过了，这里再次确认并排序)
        valid_he = [n for n in he if n not in removed_nodes]

        # 移除只有一个节点的超边
        if len(valid_he) > 1:
            final_hyperedges.append(sorted(valid_he))


    # 直接返回结果
    return final_nodes,  merge_records, mod_stats