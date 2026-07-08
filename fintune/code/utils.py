# 缩减算法，包括节点合并和超边缩减
import copy
import time
import random
from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
import gc
from typing import Any
class IncrementalJaccardCalculator:
    """
    增量Jaccard计算器 - 支持局部更新 (优化版)
    """

    def __init__(self, hyperedge_dict):
        self.hyperedge_dict = hyperedge_dict

        # 1. 节点映射优化：使用 np.int32 节省空间
        all_nodes = set()
        for hedge in hyperedge_dict.values():
            all_nodes.update(hedge)
        self.idx_to_node = np.array(list(all_nodes), dtype=np.int32)
        self.node_to_idx = {node: i for i, node in enumerate(self.idx_to_node)}

        self.incidence_matrix = None
        self._build_matrices()

        # 2. 内存优化：存储节点度数数组 (np.int32)
        self.node_degrees = self.incidence_matrix.getnnz(axis=1).astype(np.int32)

        # 3. 关键改进：用稀疏矩阵代替字典缓存交集
        # 初始化为全零稀疏矩阵，LIL格式方便修改
        n_nodes = len(self.idx_to_node)
        self.intersection_matrix = lil_matrix((n_nodes, n_nodes), dtype=np.int32)

        # 4. 初始化交集矩阵（只计算初始交集）
        print("初始化交集矩阵...")
        A = self.incidence_matrix
        self.intersection_matrix = (A @ A.T).tolil()

        # 5. 跟踪边的状态
        self.edge_states = {edge_id: True for edge_id in hyperedge_dict.keys()}

        # 6. 跟踪变更，延迟计算
        self.pending_updates = defaultdict(list)  # 记录待更新的节点对
        self.dirty_nodes = set()  # 标记需要重新计算的节点

    def _build_matrices(self):
        """构建节点-超边关联矩阵（改进版）"""
        if not self.hyperedge_dict:
            print("警告：超边字典为空，跳过矩阵构建")
            return

        print("构建稀疏矩阵...")

        # 检查空超边并记录
        empty_edges = [eid for eid, nodes in self.hyperedge_dict.items() if not nodes]
        if empty_edges:
            print(f"警告：发现 {len(empty_edges)} 个空超边，将被忽略")

        # 构建节点映射（过滤空超边）
        all_nodes = set()
        for edge in self.hyperedge_dict.values():
            if edge:  # 跳过空超边
                all_nodes.update(edge)

        if not all_nodes:
            print("错误：未找到有效节点")
            return

        n_nodes = len(self.idx_to_node)
        n_edges = len(self.hyperedge_dict) - len(empty_edges)

        # 使用生成器构建坐标
        rows = []
        cols = []
        for edge_idx, (edge_id, nodes) in enumerate(self.hyperedge_dict.items()):
            if not nodes:
                continue

            unique_nodes = set(nodes)  # 节点去重
            for node in unique_nodes:
                rows.append(self.node_to_idx[node])
                cols.append(edge_idx)

        # 构建稀疏矩阵
        data = np.ones(len(rows))
        incidence_coo = coo_matrix((data, (rows, cols)),
                                   shape=(n_nodes, n_edges))
        self.incidence_matrix = incidence_coo.tocsr()

        # 计算节点度（高效方法）
        self.node_degrees = self.incidence_matrix.getnnz(axis=1).astype(np.int32)

        print(f"矩阵构建完成: {n_nodes} 节点, {n_edges} 超边")

    def compute_initial_jaccard(self):
        """计算初始Jaccard距离矩阵"""
        print("计算初始Jaccard相似度矩阵...")
        A = self.incidence_matrix

        # 计算共同邻居数
        intersection = A.dot(A.T)

        # 转换为COO格式处理非零元素
        intersection_coo = intersection.tocoo()
        # 过滤掉行=列的对角线元素
        non_diag_mask = intersection_coo.row != intersection_coo.col
        intersection_coo.row = intersection_coo.row[non_diag_mask]
        intersection_coo.col = intersection_coo.col[non_diag_mask]
        intersection_coo.data = intersection_coo.data[non_diag_mask]
        # 计算节点度向量
        deg_i = self.node_degrees.astype(np.float32)

        # 计算并集和Jaccard距离 (向量化)
        union_values = deg_i[intersection_coo.row] + deg_i[intersection_coo.col] - intersection_coo.data
        jaccard_sim_values = np.divide(intersection_coo.data, union_values,
                                       out=np.zeros_like(intersection_coo.data, dtype=np.float32),
                                       where=union_values != 0)

        # 处理对角线
        diag_rows = np.arange(len(self.idx_to_node))
        diag_cols = np.arange(len(self.idx_to_node))
        diag_data = np.ones(len(self.idx_to_node), dtype=np.float32)

        # 合并非对角线与对角线数据
        all_rows = np.concatenate([intersection_coo.row, diag_rows])
        all_cols = np.concatenate([intersection_coo.col, diag_cols])
        all_data = np.concatenate([jaccard_sim_values, diag_data])

        # 构建CSR矩阵（内存高效）
        jaccard_dist_csr = coo_matrix((all_data, (all_rows, all_cols)),
                                      shape=intersection.shape,
                                      dtype=np.float32).tocsr()

        print(f"Jaccard矩阵构建完成，非零元素: {jaccard_dist_csr.nnz}")

        # 清理中间变量
        del intersection, intersection_coo
        gc.collect()

        return jaccard_dist_csr

    def update_for_edge_change(self, edge_id, nodes, is_removal=False):
        """
        增量更新：处理单条边的添加或删除 (优化版)
        只更新超边内部节点对的交集，避免遍历全量节点

        """
        affected_nodes = set()

        # 1. 状态检查
        target_state = not is_removal
        current_state = self.edge_states.get(edge_id, True)
        if current_state == target_state:
            return affected_nodes

        # 2. 准备工作：获取节点索引
        node_indices = []
        for node in nodes:
            if node in self.node_to_idx:
                idx = self.node_to_idx[node]
                node_indices.append(idx)
                affected_nodes.add(idx)

        if not node_indices:
            print(f"警告: 边 {edge_id} 没有有效的节点索引，无法更新。")
            self.edge_states[edge_id] = target_state
            return affected_nodes

        """增量更新：复杂度仅与超边大小k有关，O(k²)，与总节点数N无关"""
        delta = -1 if is_removal else 1

        # 3. 更新节点度
        for idx in node_indices:
            self.node_degrees[idx] += delta

        # 4. 局部更新交集矩阵 - 只更新超边内节点对 (Clique更新)
        # 对于k个节点的超边，最多有k*(k-1)/2个节点对
        n = len(node_indices)
        if n > 1:  # 只有至少2个节点才需要更新交集
            for i in range(n):
                u = node_indices[i]
                for j in range(i + 1, n):
                    v = node_indices[j]
                    # 更新交集矩阵
                    current_val = self.intersection_matrix[u, v]
                    self.intersection_matrix[u, v] = max(0, current_val + delta)
                    self.intersection_matrix[v, u] = max(0, current_val + delta)

        # 5. 更新边状态
        self.edge_states[edge_id] = target_state

        # 6. 标记受影响的节点为脏数据
        for idx in node_indices:
            self.dirty_nodes.add(idx)

        return affected_nodes

    def get_jaccard_row(self, node_idx):
        """
        核心改进：按需计算单行 Jaccard 向量
        使用稀疏矩阵和向量化操作，内存占用极低
        """
        # 获取第 u 行与其他所有节点的交集计数
        inter_row = self.intersection_matrix.getrow(node_idx)

        # 转换为CSR格式以便快速访问
        if not isinstance(inter_row, csr_matrix):
            inter_row = inter_row.tocsr()

        deg_u = self.node_degrees[node_idx]
        deg_v = self.node_degrees

        # 计算并集和Jaccard距离 (向量化)
        # 对于稀疏矩阵，我们只处理非零元素
        rows, cols = inter_row.nonzero()
        inter_data = inter_row.data

        if len(inter_data) == 0:
            return np.zeros(len(self.idx_to_node), dtype=np.float32)

        # 计算并集: deg_u + deg_v[col] - intersection
        union_data = deg_u + deg_v[cols] - inter_data

        # 计算Jaccard相似度
        jaccard_data = np.zeros_like(inter_data, dtype=np.float32)
        mask = union_data > 0
        jaccard_data[mask] = inter_data[mask] / union_data[mask]

        # 构建完整的Jaccard行向量
        jaccard_row = np.zeros(len(self.idx_to_node), dtype=np.float32)
        jaccard_row[cols] = jaccard_data
        jaccard_row[node_idx] = 1.0  # 对角线设为1

        return jaccard_row

    def get_current_jaccard_matrix(self):
        """
        获取当前Jaccard矩阵（按需计算，内存高效）
        对于大规模数据，不推荐计算完整矩阵
        """
        print("警告：获取完整Jaccard矩阵可能消耗大量内存，推荐使用get_jaccard_row")
        n_nodes = len(self.idx_to_node)

        # 使用稀疏矩阵构建
        rows = []
        cols = []
        data = []

        # 只处理有交集的节点对
        inter_coo = self.intersection_matrix.tocoo()

        # deg_i = self.node_degrees.astype(np.float32)

        # 计算并集和Jaccard
        # for i in range(len(inter_coo.row)):
        #     u, v = inter_coo.row[i], inter_coo.col[i]
        #     inter = inter_coo.data[i]
        #     union = deg_i[u] + deg_i[v] - inter
        #
        #     if union > 0:
        #         jaccard = inter / union
        #         rows.append(u)
        #         cols.append(v)
        #         data.append(jaccard)

        # 1. 处理交集矩阵：先过滤对角线元素，从根源避免重复
        inter_coo = self.intersection_matrix.tocoo()
        # 生成非对角线索引掩码（行索引 != 列索引）
        non_diag_mask = inter_coo.row != inter_coo.col
        # 过滤掉所有对角线元素，仅保留非对角线的节点对交集
        inter_coo.row = inter_coo.row[non_diag_mask]
        inter_coo.col = inter_coo.col[non_diag_mask]
        inter_coo.data = inter_coo.data[non_diag_mask]

        # 2. 向量化计算非对角线节点对的Jaccard相似度（无重复风险）
        if len(inter_coo.row) > 0:  # 存在非零非对角线元素时才计算
            deg_i = self.node_degrees.astype(np.float32)

            # 向量化运算（高效无循环）
            u = inter_coo.row
            v = inter_coo.col
            inter = inter_coo.data.astype(np.float32)
            union = deg_i[u] + deg_i[v] - inter

            # 过滤有效并集，避免除零错误
            valid_mask = union > 0
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            jaccard_valid = inter[valid_mask] / union[valid_mask]

            # 将numpy数组转为列表，批量添加（高效无冗余）
            rows.extend(u_valid.tolist())
            cols.extend(v_valid.tolist())
            data.extend(jaccard_valid.tolist())

        # 3. 手动添加对角线元素（此时无重复，因为已过滤交集矩阵的对角线）
        # 对角线元素对应节点自身相似度，固定为1.0，无叠加风险
        diag_rows = np.arange(n_nodes)
        diag_cols = np.arange(n_nodes)
        diag_data = np.ones(n_nodes, dtype=np.float32)

        # 批量添加对角线（比for循环append更高效）
        rows.extend(diag_rows.tolist())
        cols.extend(diag_cols.tolist())
        data.extend(diag_data.tolist())

        # 4. 构建稀疏矩阵（CSR格式，内存高效，支持快速行访问）
        jaccard_csr = coo_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        ).tocsr()

        # 验证：对角线元素是否唯一（可选，用于调试）
        diag_values = []
        for i in range(n_nodes):
            diag_values.append(jaccard_csr[i, i])
        print(f"对角线元素取值：{np.unique(diag_values)}")  # 应输出 [1.0]
        print(f"Jaccard矩阵构建完成，非零元素: {jaccard_csr.nnz}")

        return jaccard_csr

class KLDivergenceCalculator:
    """高效的KL散度计算器 - 支持局部更新 (优化版)"""

    def __init__(self, original_jaccard, node_to_idx):
        self.original_jaccard = original_jaccard
        self.current_jaccard_calculator = None
        self.node_to_idx = node_to_idx

        # 缓存节点的KL散度
        self.kl_cache = {}

        # 预计算原始Jaccard的行向量（稀疏存储）
        self.original_rows = {}

    def set_current_calculator(self, calculator):
        """设置当前的Jaccard计算器"""
        self.current_jaccard_calculator = calculator
        self.kl_cache.clear()
        self.original_rows.clear()

    def get_original_row(self, node_idx):
        """获取原始Jaccard行向量（带缓存）"""
        if node_idx not in self.original_rows:
            if hasattr(self.original_jaccard, 'getrow'):
                row = self.original_jaccard.getrow(node_idx)
                # 转换为稠密向量（仅对于小规模数据）
                if row.shape[1] < 10000:  # 限制大小
                    self.original_rows[node_idx] = row.toarray().flatten()
                else:
                    # 对于大规模数据，保持稀疏
                    self.original_rows[node_idx] = row
            else:
                self.original_rows[node_idx] = self.original_jaccard[node_idx, :].flatten()
        return self.original_rows[node_idx]

    # def compute_kl_for_node(self, node_idx, current_row=None):
    #     """
    #     计算单个节点的KL散度
    #     Args:
    #         node_idx: 节点索引
    #         current_row: 当前Jaccard行向量（如果为None则从计算器获取）
    #     """
    #     # 获取原始行向量
    #     original_row = self.get_original_row(node_idx)
    #
    #     # 获取当前行向量
    #     if current_row is None:
    #         if self.current_jaccard_calculator:
    #             current_row = self.current_jaccard_calculator.get_jaccard_row(node_idx)
    #         else:
    #             return float('inf')
    #
    #     # 确保两个向量都是稠密格式
    #     if sparse.issparse(original_row):
    #         original_row = original_row.toarray().flatten()
    #     if sparse.issparse(current_row):
    #         current_row = current_row.toarray().flatten()
    #
    #     # 计算KL散度
    #     eps = 1e-10
    #     p = original_row + eps
    #     q = current_row + eps
    #
    #     # 归一化
    #     p = p / (np.sum(p) + eps)
    #     q = q / (np.sum(q) + eps)
    #
    #     kl = np.sum(p * np.log(p / q))
    #     return kl

    def update_after_edge_swap(self, restore_edge_id, remove_edge_id, hyperedge_dict):
        """
        边交换后立即同步状态
        """
        # 清除受影响节点的缓存
        affected_nodes = set()

        if restore_edge_id in hyperedge_dict:
            nodes_restore = hyperedge_dict[restore_edge_id]
            for node in nodes_restore:
                if node in self.node_to_idx:
                    affected_nodes.add(self.node_to_idx[node])

        if remove_edge_id in hyperedge_dict:
            nodes_remove = hyperedge_dict[remove_edge_id]
            for node in nodes_remove:
                if node in self.node_to_idx:
                    affected_nodes.add(self.node_to_idx[node])

        # 清除缓存
        for node_idx in affected_nodes:
            if node_idx in self.kl_cache:
                del self.kl_cache[node_idx]

# --------------------辅助函数优化版----------------------------
def compute_kl_between_rows(p_row, q_row):
    """
    向量化计算两个 Jaccard 概率分布之间的 KL 散度 (优化版)
    """
    eps = 1e-10
    # 归一化为概率分布
    p_sum = np.sum(p_row) + eps
    q_sum = np.sum(q_row) + eps

    p = p_row / p_sum + eps
    q = q_row / q_sum + eps

    # 向量化计算
    return np.sum(p * np.log(p / q))

def try_edge_swap_optimization_kl_fast(node, removed_candidates, current_candidates,
                                      current_hyperedges, hyperedge_dict, removed_edges,
                                      kl_calculator, kl_current,jaccard_calc):
    """
    使用KL散度评估的边交换优化（快速版）
    避免深拷贝，使用向量化操作
    """
    start_time = time.time()
    if node not in kl_calculator.node_to_idx:
        return False

    node_idx = kl_calculator.node_to_idx[node]
    jaccard_calculator = kl_calculator.current_jaccard_calculator

    if jaccard_calculator is None:
        return False

    # 获取原始Jaccard行向量
    original_row = kl_calculator.get_original_row(node_idx)
    if sparse.issparse(original_row):
        original_row = original_row.toarray().flatten()

    n_nodes = len(jaccard_calculator.idx_to_node)
    best_restore_edge = None
    best_restore_improvement = -float('inf')
    best_remove_edge = None
    best_remove_improvement = -float('inf')

    # print(f"寻找最佳交换边 (节点 {node})...")

    # --- 寻找最佳恢复边 ---
    for restore_edge in removed_candidates:
        if restore_edge not in hyperedge_dict:
            continue

        nodes_in_edge = hyperedge_dict[restore_edge]
        node_indices = [kl_calculator.node_to_idx[n] for n in nodes_in_edge
                       if n in kl_calculator.node_to_idx]

        if not node_indices:
            continue

        # 1. 模拟恢复操作
        delta = 1
        deg_u = jaccard_calculator.node_degrees[node_idx] + delta

        # 创建当前节点的交集向量副本
        inter_row = jaccard_calculator.intersection_matrix.getrow(node_idx).toarray().flatten()

        # 更新交集
        for idx in node_indices:
            if idx != node_idx:
                inter_row[idx] += delta

        # 计算新的Jaccard行向量
        current_row = np.zeros(n_nodes, dtype=np.float32)
        current_row[node_idx] = 1.0

        # 向量化计算Jaccard
        deg_v = jaccard_calculator.node_degrees
        union = deg_u + deg_v - inter_row
        mask = union > 0
        current_row[mask] = inter_row[mask] / union[mask]

        # 计算KL散度
        kl_after = compute_kl_between_rows(original_row, current_row)
        improvement = kl_current - kl_after

        if improvement > best_restore_improvement:
            best_restore_improvement = improvement
            best_restore_edge = restore_edge

    # --- 寻找最佳删除边 ---
    for remove_edge in current_candidates:
        if remove_edge not in hyperedge_dict:
            continue

        nodes_in_edge = hyperedge_dict[remove_edge]
        node_indices = [kl_calculator.node_to_idx[n] for n in nodes_in_edge
                       if n in kl_calculator.node_to_idx]

        if not node_indices:
            continue

        # 1. 模拟删除操作
        delta = -1
        deg_u = jaccard_calculator.node_degrees[node_idx] + delta

        # 创建当前节点的交集向量副本
        inter_row = jaccard_calculator.intersection_matrix.getrow(node_idx).toarray().flatten()

        # 更新交集
        for idx in node_indices:
            if idx != node_idx:
                inter_row[idx] = max(0, inter_row[idx] + delta)

        # 计算新的Jaccard行向量
        current_row = np.zeros(n_nodes, dtype=np.float32)
        current_row[node_idx] = 1.0

        # 向量化计算Jaccard
        deg_v = jaccard_calculator.node_degrees
        union = deg_u + deg_v - inter_row
        mask = union > 0
        current_row[mask] = inter_row[mask] / union[mask]

        # 计算KL散度
        kl_after = compute_kl_between_rows(original_row, current_row)
        improvement = kl_current - kl_after

        if improvement > best_remove_improvement:
            best_remove_improvement = improvement
            best_remove_edge = remove_edge

    # --- 执行交换操作 ---
    if (best_remove_improvement + best_restore_improvement) > 0:
        if best_restore_edge and best_remove_edge:
            # 执行交换
            current_hyperedges[best_restore_edge] = hyperedge_dict[best_restore_edge]
            if best_remove_edge in current_hyperedges:
                del current_hyperedges[best_remove_edge]

            # 更新已删除边列表
            removed_edges.remove(best_restore_edge)
            removed_edges.append(best_remove_edge)

            # 立即同步状态
            kl_calculator.update_after_edge_swap(best_restore_edge, best_remove_edge, hyperedge_dict)
            nodes_in_edge_restore = hyperedge_dict[best_restore_edge]
            jaccard_calc.update_for_edge_change(best_restore_edge, nodes_in_edge_restore, is_removal=False)
            nodes_in_edge_remove = hyperedge_dict[best_restore_edge]
            jaccard_calc.update_for_edge_change(best_remove_edge, nodes_in_edge_remove, is_removal=True)
            print(f"边交换优化: 恢复 {best_restore_edge}(ΔKL={best_restore_improvement:.4f}), "
                  f"删除 {best_remove_edge}(ΔKL={best_remove_improvement:.4f})")
            return True
    end_time = time.time()
    print(f"微调该节点，耗时: {end_time - start_time:.2f}s")
    return False

def compute_group_changes(original_groups, current_groups):
    """计算分组分布的变化"""
    changes = {}
    all_groups = set(original_groups.keys()) | set(current_groups.keys())

    for group in all_groups:
        original_count = len(original_groups.get(group, []))
        current_count = len(current_groups.get(group, []))
        changes[group] = current_count - original_count

    return changes


def compute_group_distribution(hyperedge_dict):
    """
    计算超边的分组分布（只按超边大小分组）
    Returns:
        group_dist: dict，分组键→边数量
        edge_to_group: dict，边ID→分组键
        group_to_edges: dict，分组键→边ID列表
    """
    edge_to_group = {}  # 边ID→分组键
    group_to_edges = defaultdict(list)  # 分组→边列表

    for edge_id, nodes in hyperedge_dict.items():
        # 计算超边大小
        hsize = len(nodes)

        # 只按超边大小分组，创建分组键 [hsize]
        group_key = f"{hsize}"  # 直接使用大小作为分组键

        # 存储分组信息
        edge_to_group[edge_id] = group_key
        group_to_edges[group_key].append(edge_id)

    # 计算分组分布
    group_dist = {group: len(edges) for group, edges in group_to_edges.items()}

    return group_dist, edge_to_group, group_to_edges

def optimize_group_distribution(
        current_hyperedges,
        original_hyperedge_dict,
        removed_edges,
        original_groups,  # 这里实际是 original_group_to_edges（分组→边列表）
        random_state,
        jaccard_calc,
        kl_calculator,
        original_edge_to_group  # 边→分组映射（用于查询边所属分组）
):
    """优化分组分布（已补充边收集和状态同步）"""
    print("开始分组分布优化...")

    # --------------------------
    # 1. 计算当前分组分布（注意：compute_group_distribution需按之前修复的返回值调整）
    # --------------------------
    # 调用修复后的compute_group_distribution，获取 (分组→数量, 边→分组, 分组→边列表)
    current_group_dist, current_edge_to_group, current_group_to_edges = compute_group_distribution(current_hyperedges)

    # 计算分组变化（对比原始分组和当前分组）
    group_changes = compute_group_changes(original_groups, current_group_to_edges)
    # max_change_group = find_max_group_change(group_changes)
    #
    # print(f"分组分布变化: {group_changes}")
    # print(f"变化最大的分组: {max_change_group}")
    #
    # with open(output_file, 'a', encoding='utf-8') as f:
    #     f.write(f"分组分布变化: {group_changes}\n")
    #     f.write(f"变化最大的分组: {max_change_group}\n")

    # --------------------------
    # 2. 计算目标分组数量（修改：优先分配大超边）
    # --------------------------
    total_original = sum(len(edges) for edges in original_groups.values())
    total_current = total_original - len(removed_edges)

    # 计算原始分组概率
    target_probs = {}
    for group, edges in original_groups.items():
        target_probs[group] = len(edges) / total_original

    # 修改1：按超边大小降序排列分组（大超边在前）
    sorted_groups = sorted(target_probs.keys(), key=lambda x: int(x), reverse=True)

    # 修改2：按超边大小降序分配目标边数
    target_counts = {}
    remaining_edges = total_current

    # 按原始比例分配目标边数，但按大超边优先的顺序
    for group in sorted_groups:
        if remaining_edges <= 0:
            target_counts[group] = 0
            continue

        # 计算该分组应得的目标边数
        target_edges = round(target_probs[group] * total_current)

        # 确保不超过剩余边数
        actual_target = min(target_edges, remaining_edges)
        target_counts[group] = actual_target
        remaining_edges -= actual_target

    # 如果还有剩余边数（由于四舍五入误差），加到最大的超边分组上
    if remaining_edges > 0:
        # 找最大的超边分组（已经按降序排列，第一个就是最大的）
        for group in sorted_groups:
            if remaining_edges <= 0:
                break
            target_counts[group] += 1
            remaining_edges -= 1

    # 调整总数量偏差（确保目标数量和当前超边数一致）
    actual_total = sum(target_counts.values())
    if actual_total != total_current:
        diff = total_current - actual_total

        # 修改3：调整偏差时优先保证大超边
        if diff > 0:  # 需要增加边数，加到大超边上
            # 按超边大小降序排列，优先加到大超边
            sorted_adjust = sorted(target_counts.keys(), key=lambda x: int(x), reverse=True)
            for group in sorted_adjust:
                if diff <= 0:
                    break
                target_counts[group] += 1
                diff -= 1
        else:  # 需要减少边数，从小超边开始减
            # 按超边大小升序排列，优先减小超边
            sorted_adjust = sorted(target_counts.keys(), key=lambda x: int(x))
            for group in sorted_adjust:
                if diff >= 0:
                    break
                if target_counts[group] > 0:
                    target_counts[group] -= 1
                    diff += 1

    # 计算每个分组需要调整的数量（目标-当前）
    adjustment_needed = {}
    for group in set(original_groups.keys()) | set(current_group_to_edges.keys()):
        current_count = len(current_group_to_edges.get(group, []))
        target_count = target_counts.get(group, 0)
        adjustment_needed[group] = target_count - current_count

    print(f"目标边数: {target_counts}")
    print(f"当前边数: {current_group_dist}")
    print(f"需要调整的数量: {adjustment_needed}")

    # --------------------------
    # 3. 核心修复：创建列表收集恢复/删除的边ID
    # --------------------------
    restored_edges_list = []  # 收集所有"恢复的边"
    removed_in_adjustment_list = []  # 收集所有"分组优化删除的边"

    # --------------------------
    # 4. 恢复边（调整正偏差分组）
    # --------------------------
    restored_edges = 0

    # 修改4：恢复边时优先恢复大超边
    # 按超边大小降序排列需要恢复的分组
    restore_groups = sorted([(group, adj) for group, adj in adjustment_needed.items() if adj > 0],
                            key=lambda x: int(x[0]), reverse=True)

    for group, adjustment in restore_groups:
        # 筛选：已删除的边 + 属于当前分组
        candidates = []
        for edge in removed_edges:
            # 通过original_edge_to_group查询边的原始分组
            if edge in original_edge_to_group and original_edge_to_group[edge] == group:
                candidates.append(edge)

        if candidates:
            # 确定要恢复的边数（不超过需要调整的数量）
            restore_count = min(adjustment, len(candidates))
            to_restore = random_state.sample(candidates, restore_count)

            # 执行恢复操作
            for edge in to_restore:
                current_hyperedges[edge] = original_hyperedge_dict[edge]  # 恢复到当前超边
                removed_edges.remove(edge)  # 从已删除列表中移除
                restored_edges_list.append(edge)  # 收集恢复的边

            restored_edges += restore_count
            print(f"恢复组 {group} 的 {restore_count} 条边: {to_restore}")
            # with open(output_file, 'a', encoding='utf-8') as f:
            #     f.write(f"恢复组 {group} 的 {restore_count} 条边: {to_restore}\n")

    # --------------------------
    # 5. 删除边（调整负偏差分组）
    # --------------------------
    removed_in_adjustment = 0

    # 修改5：删除边时优先删除小超边
    # 按超边大小升序排列需要删除的分组
    remove_groups = sorted([(group, adj) for group, adj in adjustment_needed.items() if adj < 0],
                           key=lambda x: int(x[0]))

    for group, adjustment in remove_groups:
        # 筛选：当前存在的边 + 属于当前分组
        candidates = []
        for edge in current_hyperedges.keys():
            if edge in original_edge_to_group and original_edge_to_group[edge] == group:
                candidates.append(edge)

        if candidates:
            # 确定要删除的边数（不超过需要调整的数量的绝对值）
            remove_count = min(abs(adjustment), len(candidates))
            to_remove = random_state.sample(candidates, remove_count)

            # 执行删除操作
            for edge in to_remove:
                del current_hyperedges[edge]  # 从当前超边中删除
                removed_edges.append(edge)  # 加入已删除列表
                removed_in_adjustment_list.append(edge)  # 收集删除的边

            removed_in_adjustment += remove_count
            print(f"删除组 {group} 的 {remove_count} 条边: {to_remove}")
            # with open(output_file, 'a', encoding='utf-8') as f:
            #     f.write(f"删除组 {group} 的 {remove_count} 条边: {to_remove}\n")

    # --------------------------
    # 6. 输出分组优化结果（原有逻辑不变）
    # --------------------------
    final_group_dist, _, final_group_to_edges = compute_group_distribution(current_hyperedges)
    print(f"最终分组分布: {final_group_dist}")
    print(f"最终超边数量: {len(current_hyperedges)}")
    print(f"恢复的边数: {restored_edges}, 删除的边数: {removed_in_adjustment}")

    # --------------------------
    # 7. 状态同步：更新Jaccard计算器和KL缓存（核心修复）
    # --------------------------
    # 7.1 同步"恢复的边"：更新Jaccard交并集/节点度，清空KL缓存
    for edge in restored_edges_list:
        if edge not in original_hyperedge_dict:
            print(f"警告：恢复的边 {edge} 不在原始超边字典中，跳过同步")
            continue
        nodes_in_edge = original_hyperedge_dict[edge]
        # 调用Jaccard计算器的增量更新（添加边）
        jaccard_calc.update_for_edge_change(edge, nodes_in_edge, is_removal=False)
        # 清空该边涉及节点的KL缓存（避免使用旧值）
        for node in nodes_in_edge:
            if node in kl_calculator.node_to_idx:
                node_idx = kl_calculator.node_to_idx[node]
                kl_calculator.kl_cache.pop(node_idx, None)  # 安全删除，不存在则忽略

    # 7.2 同步"删除的边"：更新Jaccard交并集/节点度，清空KL缓存
    for edge in removed_in_adjustment_list:
        if edge not in original_hyperedge_dict:
            print(f"警告：删除的边 {edge} 不在原始超边字典中，跳过同步")
            continue
        nodes_in_edge = original_hyperedge_dict[edge]
        # 调用Jaccard计算器的增量更新（删除边）
        jaccard_calc.update_for_edge_change(edge, nodes_in_edge, is_removal=True)
        # 清空该边涉及节点的KL缓存
        for node in nodes_in_edge:
            if node in kl_calculator.node_to_idx:
                node_idx = kl_calculator.node_to_idx[node]
                kl_calculator.kl_cache.pop(node_idx, None)

    # --------------------------
    # 8. 返回更新后的结果和收集的边列表
    # --------------------------
    return (
        current_hyperedges,
        removed_edges,
        restored_edges_list,
        removed_in_adjustment_list
    )

# ===== 新增辅助函数 =====
def compute_current_average_kl(kl_calculator,KL, sample_nodes=None, sample_size=100):
    """
    计算当前平均KL散度
    """
    all_nodes = list(kl_calculator.node_to_idx.keys())

    # 如果没有提供采样节点，随机采样
    if sample_nodes is None or len(sample_nodes) == 0:
        sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))

    total_kl = 0
    count = 0

    for node in sample_nodes:
        node_idx = kl_calculator.node_to_idx.get(node)
        if node_idx is not None:
            kl = KL[node_idx]
            total_kl += kl
            count += 1

    return total_kl / count if count > 0 else 0

def compute_distribution_distance(current_dist, original_dist, use_normalization=True):
    """
    计算两个分组分布之间的距离（先归一化再计算）

    参数:
        current_dist: 当前分组分布字典 {group_id: count}
        original_dist: 原始分组分布字典 {group_id: count}
        use_normalization: 是否先归一化（默认True）

    返回:
        分布距离（欧氏距离）
    """
    if not current_dist or not original_dist:
        return float('inf')

    # 获取所有分组ID的并集
    all_groups = set(list(current_dist.keys()) + list(original_dist.keys()))

    if use_normalization:
        # 步骤1：归一化两个分布（转换为概率分布）
        current_total = sum(current_dist.values())
        original_total = sum(original_dist.values())

        # 避免除零错误
        if current_total == 0 or original_total == 0:
            return float('inf')

        current_prob = {}
        original_prob = {}

        for group in current_dist:
            current_prob[group] = current_dist[group] / current_total

        for group in original_dist:
            original_prob[group] = original_dist[group] / original_total

        # 步骤2：计算归一化后的欧氏距离
        distance = 0
        for group in all_groups:
            current_val = current_prob.get(group, 0)
            original_val = original_prob.get(group, 0)
            distance += (current_val - original_val) ** 2

        return distance ** 0.5
    else:
        # 如果不归一化，直接计算原始计数的距离（不推荐）
        distance = 0
        for group in all_groups:
            current_val = current_dist.get(group, 0)
            original_val = original_dist.get(group, 0)
            distance += (current_val - original_val) ** 2

        return distance ** 0.5


def compute_solution_mmd(kl_divergence, group_distance, kl_min, kl_max, group_min, group_max):
    """
    计算单个解的MMD值

    参数:
        kl_divergence: KL散度值
        group_distance: 分组距离值
        kl_min, kl_max: KL散度的最小值和最大值
        group_min, group_max: 分组距离的最小值和最大值

    返回:
        MMD值
    """
    # 使用公式2计算标准化值
    if kl_max > kl_min:
        f1_prime = (kl_divergence - kl_max) / (kl_min - kl_max)
    else:
        f1_prime = 0.5  # 所有值相等时的默认值

    if group_max > group_min:
        f2_prime = (group_distance - group_max) / (group_min - group_max)
    else:
        f2_prime = 0.5

    # 使用公式3计算MMD
    mmd = f1_prime + f2_prime
    return mmd


def compute_evaluation_metrics_new(original_hyperedges, simplified_hyperedges, k_eigenvalues=50):
    """
    计算简化超图与原始超图之间的评估指标
    """
    # 将超边字典转换为列表
    original_edges = list(original_hyperedges.values())
    simplified_edges = list(simplified_hyperedges.values())

    # 计算节点度分布
    def compute_degree_distribution(hyperedges):
        node_degree = defaultdict(int)
        for edge in hyperedges:
            for node in edge:
                node_degree[node] += 1

        if not node_degree:
            return np.array([]), np.array([]), 0

        degrees = list(node_degree.values())
        max_degree = max(degrees)
        hist = np.zeros(max_degree + 1)
        for d in degrees:
            hist[d] += 1
        prob_dist = hist / len(degrees)
        return np.array(degrees), prob_dist, max_degree

    # 计算超边大小分布
    def compute_size_distribution(hyperedges):
        if not hyperedges:
            return np.array([]), np.array([]), 0

        sizes = [len(edge) for edge in hyperedges]
        max_size = max(sizes)
        hist = np.zeros(max_size + 1)
        for s in sizes:
            if s < len(hist):
                hist[s] += 1
        prob_dist = hist / len(sizes)
        return np.array(sizes), prob_dist, max_size

    # 计算KL散度
    def kl_divergence(p, q):
        epsilon = 1e-10
        kl = 0.0
        for i in range(len(p)):
            p_i = max(p[i], epsilon)
            q_i = max(q[i] if i < len(q) else epsilon, epsilon)
            kl += p_i * np.log(p_i / q_i)
        return kl

    def compute_volume_density(hyperedges):
        """计算超图的volume density ρ(S)"""
        if not hyperedges:
            return 0.0

        # 收集所有节点
        nodes = set()
        for edge in hyperedges:
            nodes.update(edge)

        # 如果没有节点，返回0
        if not nodes:
            return 0.0

        # 构建邻接字典：记录每个节点的邻居集合
        neighbors = defaultdict(set)
        for edge in hyperedges:
            for node in edge:
                # 添加该超边中除自身外的所有节点作为邻居
                neighbors[node].update(edge)
                neighbors[node].discard(node)  # 移除自身

        # 计算所有节点的邻居数量之和
        total_neighbors = sum(len(neighbors[node]) for node in nodes)

        # 计算volume density
        rho = total_neighbors / len(nodes)
        return rho

    def compute_neighbor_jaccard_similarity(original_edges, simplified_edges):
        """
        计算平均节点邻域Jaccard相似度（邻接保持率）
        """

        def get_node_neighbors(hyperedges):
            """获取每个节点的邻居集合"""
            neighbors = defaultdict(set)
            for edge in hyperedges:
                for node in edge:
                    # 添加该超边中除自身外的所有节点作为邻居
                    neighbors[node].update(edge)
                    neighbors[node].discard(node)  # 移除自身
            return neighbors

        original_neighbors = get_node_neighbors(original_edges)
        simplified_neighbors = get_node_neighbors(simplified_edges)

        # 只考虑在两个图中都出现的节点
        common_nodes = set(original_neighbors.keys()) & set(simplified_neighbors.keys())

        if not common_nodes:
            return 0.0

        jaccard_similarities = []
        for node in common_nodes:
            orig_neigh = original_neighbors[node]
            simp_neigh = simplified_neighbors[node]

            if not orig_neigh and not simp_neigh:
                similarity = 1.0
            elif not orig_neigh or not simp_neigh:
                similarity = 0.0
            else:
                intersection = len(orig_neigh & simp_neigh)
                union = len(orig_neigh | simp_neigh)
                similarity = intersection / union if union > 0 else 0.0

            jaccard_similarities.append(similarity)

        return np.mean(jaccard_similarities)

    # 计算原始和简化超图的分布
    degrees_o, p_o, dmax_o = compute_degree_distribution(original_edges)
    degrees_s, p_s, dmax_s = compute_degree_distribution(simplified_edges)

    sizes_o, q_o, cmax_o = compute_size_distribution(original_edges)
    sizes_s, q_s, cmax_s = compute_size_distribution(simplified_edges)

    # 对齐分布向量长度
    max_degree = max(dmax_o, dmax_s)
    p_o_full = np.zeros(max_degree + 1)
    p_s_full = np.zeros(max_degree + 1)
    p_o_full[:len(p_o)] = p_o
    p_s_full[:len(p_s)] = p_s

    max_size = max(cmax_o, cmax_s)
    q_o_full = np.zeros(max_size + 1)
    q_s_full = np.zeros(max_size + 1)
    q_o_full[:len(q_o)] = q_o
    q_s_full[:len(q_s)] = q_s

    # 1. 平均节点度相对误差 (RE-d)
    avg_degree_o = np.mean(degrees_o) if degrees_o.size > 0 else 0
    avg_degree_s = np.mean(degrees_s) if degrees_s.size > 0 else 0
    re_d = abs(avg_degree_s - avg_degree_o) / avg_degree_o if avg_degree_o > 0 else float('inf')

    # 2. 平均超边大小相对误差 (RE-s)
    avg_size_o = np.mean(sizes_o) if sizes_o.size > 0 else 0
    avg_size_s = np.mean(sizes_s) if sizes_s.size > 0 else 0
    re_s = abs(avg_size_s - avg_size_o) / avg_size_o if avg_size_o > 0 else float('inf')

    # 3. 节点度分布L1误差 (L1-d)
    l1_d = np.sum(np.abs(p_s_full - p_o_full))

    # 4. 超边大小分布L1误差 (L1-s)
    l1_s = np.sum(np.abs(q_s_full[1:] - q_o_full[1:]))  # 忽略大小为0

    # 5. 节点度分布KL散度 (KL-d)
    kl_d = kl_divergence(p_o_full, p_s_full)

    # 6. 超边大小分布KL散度 (KL-s)
    kl_s = kl_divergence(q_o_full[1:], q_s_full[1:])  # 忽略大小为0

    # 7. 原始超图的volume density
    rho_original = compute_volume_density(original_edges)

    # 8. 简化超图的volume density
    rho_simplified = compute_volume_density(simplified_edges)

    # 9. volume density相对误差
    re_rho = abs(rho_simplified - rho_original) / rho_original if rho_original != 0 else float('inf')

    # 10. 平均节点邻域Jaccard相似度 (邻接保持率)
    neighbor_similarity = compute_neighbor_jaccard_similarity(original_edges, simplified_edges)

    # 返回所有指标
    return {
        'RE-d': re_d,
        'RE-s': re_s,
        'L1-d': l1_d,
        'L1-s': l1_s,
        'KL-d': kl_d,
        'KL-s': kl_s,
        'RE-ρ': re_rho,
        'NR': neighbor_similarity,  # Neighbor Retention
    }


def fast_kl_divergence(A, B, eps=1e-10):
    # 1. 确保 CSR 格式
    A = A.tocsr()
    B = B.tocsr()

    # 2. 矢量化归一化函数
    def get_normed_data(mat):
        row_sums = np.array(mat.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        # 获取每个非零元素所属的行索引 (例如 [0,0,0,1,1,2...])
        row_indices = np.repeat(np.arange(mat.shape[0]), np.diff(mat.indptr))
        normed_data = mat.data / row_sums[row_indices]
        return normed_data

    A_norm_data = get_normed_data(A)
    # B 这里我们需要保留矩阵对象，因为要索引它
    B_row_sums = np.array(B.sum(axis=1)).ravel()
    B_row_sums[B_row_sums == 0] = 1.0

    # 3. 核心技巧：只计算 A 中非零元素对应的 B 的值
    # 直接提取 B 中与 A 的非零位 (A.indices) 对应的值
    # 虽然这步会产生一个和 A.data 等长的密集向量，但比循环快得多
    rows_of_a = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    cols_of_a = A.indices

    # 利用矩阵的高级索引批量获取 B 的对应值
    # 注意：B_norm_values[i] = B[row_a_i, col_a_i] / B_row_sum[row_a_i]
    b_vals_at_a_indices = np.array(B[rows_of_a, cols_of_a]).ravel()
    b_norm_at_a_indices = b_vals_at_a_indices / B_row_sums[rows_of_a]

    # 4. 矢量化 KL 计算
    # KL = sum( P * log(P/Q) )
    # P = A_norm_data
    # Q = b_norm_at_a_indices + eps
    p = A_norm_data
    q = b_norm_at_a_indices + eps

    # 计算每个非零元的贡献
    elementwise_kl = p * np.log((p + eps) / q)

    # 5. 按行求和还原回每个节点的 KL 散度
    # 使用 np.bincount 按行索引累加结果
    kl_per_row = np.bincount(rows_of_a, weights=elementwise_kl, minlength=A.shape[0])

    return kl_per_row


def hyperedge_sparsification_incremental_fast(hyperedge_dict, k, tune_nodes_num=50,
                                             random_state=None):
    """
    超图简化主函数 - 快速版 (针对百万节点优化)
    """
    if random_state is None:
        random_state = random.Random()

    print(f"开始快速增量计算版超图删边过程，初始超边数量: {len(hyperedge_dict)}，计划删除 {k} 条边")

    # 记录节点所属超边集合
    print("计算节点所属超边集合")
    get_incident_hyperedges = defaultdict(set)
    for hyperedge_id, nodes_in_edge in hyperedge_dict.items():
        for node in nodes_in_edge:
            get_incident_hyperedges[node].add(hyperedge_id)

    # 初始化增量Jaccard计算器
    print("初始化增量Jaccard计算器...")
    jaccard_calc = IncrementalJaccardCalculator(hyperedge_dict)

    # 计算原始Jaccard距离矩阵
    print("计算原始Jaccard距离矩阵...")
    start_origina_jaccard = time.time()
    # 存在一个问题就是对角线元素是2，自己和自己的相似度为2有问题，计算交集数时，计算了对角线的度数，赋值了一次1，后面计算交集又加了一次1
    original_jaccard = jaccard_calc.compute_initial_jaccard()
    end_origina_jaccard = time.time()
    duration_origina_jaccard = end_origina_jaccard-start_origina_jaccard
    print(f"计算原始Jaccard距离矩阵耗时{duration_origina_jaccard}")
    # 初始化KL散度计算器
    kl_calculator = KLDivergenceCalculator(original_jaccard, jaccard_calc.node_to_idx)
    kl_calculator.set_current_calculator(jaccard_calc)

    # 计算原始分组信息
    print("计算原始超边分组分布...")
    start_origina_group = time.time()
    # 计算原始分组信息，original_group_dist={"hypergraph size":对应数量}， original_edge_to_group 超边对应超边大小 original_group_to_edges 超边大小所对应的超边列表
    original_group_dist, original_edge_to_group, original_group_to_edges = compute_group_distribution(hyperedge_dict)
    end_origina_group = time.time()
    duration_origina_group = end_origina_group - start_origina_group
    print(f"计算原始超边分组耗时{duration_origina_group}")
    # 第一步：随机删除k条边
    print(f"随机删除 {k} 条边...")
    start_suiji = time.time()
    current_hyperedges = hyperedge_dict.copy()
    hyperedge_keys = list(hyperedge_dict.keys())
    random_state.shuffle(hyperedge_keys)

    removed_edges = []
    for i in range(k):
        if hyperedge_keys:
            edge_to_remove = hyperedge_keys.pop()
            removed_edges.append(edge_to_remove)
            del current_hyperedges[edge_to_remove]
            nodes_in_edge = hyperedge_dict[edge_to_remove]
            jaccard_calc.update_for_edge_change(edge_to_remove, nodes_in_edge, is_removal=True)

    end_suiji = time.time()
    print(f"随机删除了 {len(removed_edges)} 条边，耗时: {end_suiji - start_suiji:.2f}s")

    # 迭代优化
    solution_history = []
    kl_history = []
    group_distance_history = []
    iteration = 0

    while True:
        iteration += 1
        print(f"\n开始迭代 {iteration}")

        # 第一步：节点距离分布优化
        print("第一步：节点距离分布优化")

        # 选择变化最大的节点进行优化
        all_nodes = list(jaccard_calc.node_to_idx.keys())
        candidate_nodes = random_state.sample(all_nodes, min(tune_nodes_num, len(all_nodes)))

        # 批量计算KL散度
        start_current_jaccard = time.time()
        current_jaccard = jaccard_calc.get_current_jaccard_matrix()
        # node_indices = [jaccard_calc.node_to_idx[node] for node in all_nodes]
        end_current_jaccard = time.time()
        duration_current_jaccard = end_current_jaccard-start_current_jaccard
        print(f"计算当前的jaccard距离矩阵耗时{duration_current_jaccard}")
        start_vkl = time.time()

        KL = fast_kl_divergence(original_jaccard, current_jaccard)
        kl_changes = {node: KL[i] for i, node in enumerate(all_nodes)}

        # 按KL散度排序
        sorted_nodes = sorted(kl_changes.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, kl in sorted_nodes[:min(tune_nodes_num, len(sorted_nodes))]]

        end_vkl = time.time()
        print(f"批量计算所有节点的KL散度耗时: {end_vkl - start_vkl:.2f}s")
        print(f"处理前 {len(top_nodes)} 个变化最大的节点")

        # 对每个节点进行微调
        start_allvex = time.time()
        for node_idx, node in enumerate(top_nodes):
            if node_idx % 10 == 0:
                print(f"处理节点 {node_idx + 1}/{len(top_nodes)}")

            candidate_edges = get_incident_hyperedges.get(node, set())
            removed_candidates = [edge for edge in candidate_edges if edge in removed_edges]
            current_candidates = [edge for edge in candidate_edges if edge in current_hyperedges]

            if not removed_candidates or not current_candidates:
                continue

            try_edge_swap_optimization_kl_fast(
                node, removed_candidates, current_candidates,
                current_hyperedges, hyperedge_dict, removed_edges,
                kl_calculator, kl_changes[node],jaccard_calc)

        end_allvex = time.time()
        print(f"\n所有节点微调花费时间: {end_allvex - start_allvex:.2f}s")

        # 在这一步增加计算MD，一次迭代，计算两个MD，所以不能用iteration指代当前的图
        # 目标1：计算当前KL散度（平均KL散度）
        current_kl = compute_current_average_kl(kl_calculator, KL, candidate_nodes)
        kl_history.append(current_kl)

        # 目标2：计算当前分组分布距离
        current_group_dist, _, _ = compute_group_distribution(current_hyperedges)
        group_distance = compute_distribution_distance(current_group_dist, original_group_dist)
        group_distance_history.append(group_distance)

        # 创建当前解对象
        current_solution = {
            'iteration': iteration,
            'kl_divergence': current_kl,
            'group_distance': group_distance,
            'curr_hyperedge_id': list(set(current_hyperedges.keys()))
        }

        solution_history.append(current_solution)

        # 使用MMD判断收敛
        if len(solution_history) >= 6:  # 至少需要4个历史记录
            # 计算全部历史数据的MMD，找到最大的MMD对应的解，判断该解是否是当前的迭代数，以此判断循环是否结束
            # 遍历所有历史解：
            mmd_value = []
            for i in range(0, len(solution_history)):
                # 计算MMD值
                kl_min, kl_max = min(kl_history), max(kl_history)
                group_min, group_max = min(group_distance_history), max(group_distance_history)

                mmd = compute_solution_mmd(solution_history[i]["kl_divergence"], solution_history[i]["group_distance"],
                                           kl_min, kl_max, group_min, group_max)
                mmd_value.append(mmd)
            mmd_max_index = mmd_value.index(max(mmd_value))
            if mmd_max_index != len(solution_history)-1:
                print(f"\nMMD收敛条件满足！在迭代 {iteration} 停止")
                break
        # 第二步：分组分布优化
        start_group = time.time()
        print("第二步：分组分布优化")

        current_hyperedges, removed_edges, restored_edges, removed_adjust_edges = optimize_group_distribution(
            current_hyperedges, hyperedge_dict, removed_edges,
            original_group_to_edges, random_state,
            jaccard_calc, kl_calculator, original_edge_to_group
        )
        end_group = time.time()
        print(f"\n分组优化花费时间: {end_group - start_group:.2f}s")

        print("分组优化后同步状态...")

        # ===== 新增：计算当前解的两个目标值和MMD =====
        # 目标1：计算当前KL散度（平均KL散度）
        current_kl = compute_current_average_kl(kl_calculator, KL, candidate_nodes)
        kl_history.append(current_kl)

        # 目标2：计算当前分组分布距离
        current_group_dist, _, _ = compute_group_distribution(current_hyperedges)
        group_distance = compute_distribution_distance(current_group_dist, original_group_dist)
        group_distance_history.append(group_distance)

        # 创建当前解对象
        current_solution = {
            'iteration': iteration,
            'kl_divergence': current_kl,
            'group_distance': group_distance,
            'curr_hyperedge_id': list(set(current_hyperedges.keys()))
        }

        solution_history.append(current_solution)

        # 使用MMD判断收敛
        if len(solution_history) >= 6:  # 至少需要4个历史记录
            # 计算全部历史数据的MMD，找到最大的MMD对应的解，判断该解是否是当前的迭代数，以此判断循环是否结束
            # 遍历所有历史解：
            mmd_value = []
            for i in range(0, len(solution_history)):
                # 计算MMD值
                kl_min, kl_max = min(kl_history), max(kl_history)
                group_min, group_max = min(group_distance_history), max(group_distance_history)

                mmd = compute_solution_mmd(solution_history[i]["kl_divergence"], solution_history[i]["group_distance"],
                                           kl_min, kl_max, group_min, group_max)
                mmd_value.append(mmd)
            mmd_max_index = mmd_value.index(max(mmd_value))
            if mmd_max_index != len(solution_history)-1:
                print(f"\nMMD收敛条件满足！在迭代 {iteration} 停止")
                break
    print(f"\n优化完成，总迭代次数: {iteration}")

    # 从历史解中选择最佳解（MMD值最大的解）
    best_solution = solution_history[mmd_max_index]
    sampled_hypergraph = defaultdict()
    if best_solution:
        current_hyperedges_id = best_solution['curr_hyperedge_id']
        for edge in current_hyperedges_id:
            sampled_hypergraph[edge] = hyperedge_dict[edge]

    return sampled_hypergraph



def coarse_hypergraph_int_4_2hop(
        node_1hop: list[int],
        hyperedges_1hop_dict: dict[int, list[int]],
        hyperedges_2hop_dict: dict[int, list[int]],
        target_com: int,
        TARGET_KEEP_NUM: int,
        labels: list[int] | None = None
) -> tuple[list[int], list[list[int]]]:
    # 1. 创建本地副本，避免修改原始数据
    local_hyperedges_1hop = copy.deepcopy(hyperedges_1hop_dict)
    local_hyperedges_2hop = copy.deepcopy(hyperedges_2hop_dict)
    node_1hop = sorted(node_1hop)

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
    # Julia: [node_idx-1 for ... enumerate(labels)] 暗示 labels 下标对应 node_id+1
    # Python: 假设 labels 列表的下标即为 node_id
    label_nodes = set()
    if labels is not None:
        for node_idx, label_val in enumerate(labels):
            if label_val == target_com:
                label_nodes.add(node_idx)

    hyperedge_weights = {}
    # 遍历 2-hop 超边
    for hyper_id, hyperedge in local_hyperedges_2hop.items():
        hyperedge_count = len(hyperedge)
        # 计算该超边中有多少节点属于目标社区
        community_node_count = sum(1 for node in hyperedge if node in label_nodes)

        if community_node_count > 0:
            weight = community_node_count / hyperedge_count
        else:
            weight = -1.0
        hyperedge_weights[hyper_id] = weight

    # 计算超边尺寸 (degree of hyperedge)
    hyperedge_degree_map = {hid: len(he) for hid, he in local_hyperedges_2hop.items()}

    # 构建节点到超边的映射 (只针对 node_1hop 中的节点)
    node_to_hyperedges = defaultdict(list)
    for hyper_id, hyperedge in local_hyperedges_2hop.items():
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

        # # 跳过保护节点
        # if node1 in protect_set or node2 in protect_set:
        #     continue

        # 标签检查
        if labels is not None:
            # 边界检查
            idx1_valid = 0 <= node1 < len(labels)
            idx2_valid = 0 <= node2 < len(labels)

            if idx1_valid and idx2_valid:
                if labels[node1] != labels[node2]:
                    continue
            else:
                # 索引无效时的保守策略：跳过
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

    final_nodes = new_nodes

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

    # -------- 处理 2-hop 超边 --------
    # for he in hyperedges_2hop_dict.values():
    #     valid_he = [n for n in he if n not in removed_nodes]

    #     if len(valid_he) > 1:
    #         final_hyperedges.append(sorted(valid_he))

    # 只留下与 final_nodes 相关的超边
    # final_nodes_set = set(final_nodes)
    # filtered_hyperedges = []
    #
    # for hyperedge in final_hyperedges:
    #     if any(node in final_nodes_set for node in hyperedge):
    #         filtered_hyperedges.append(hyperedge)
    #
    # print(f"\n只留下与一跳节点相关的超边数量= {len(filtered_hyperedges)}")

    # 8. 超边合并 (基于 Jaccard/Overlap 相似度)
    # print("\n开始超边合并...")
    #
    # # 转换为 Set 以便计算 Jaccard
    # hyperedge_sets = [set(he) for he in filtered_hyperedges]
    # n_he = len(hyperedge_sets)
    #
    # hyperedge_similarities = []
    # for i in range(n_he):
    #     set_i = hyperedge_sets[i]
    #     for j in range(i + 1, n_he):
    #         set_j = hyperedge_sets[j]
    #
    #         intersection_len = len(set_i.intersection(set_j))
    #         # Julia 代码使用的是 min(len, len) 作为分母，这是 Overlap Coefficient，但变量名叫 jaccard
    #         union_len = min(len(set_i), len(set_j))
    #
    #         jaccard = intersection_len / union_len if union_len > 0 else 0.0
    #
    #         hyperedge_similarities.append({
    #             'indices': (i, j),
    #             'jaccard': jaccard,
    #             'hyperedge_i': list(set_i),
    #             'hyperedge_j': list(set_j)
    #         })
    #
    # # 按 Jaccard 降序排序
    # hyperedge_similarities.sort(key=lambda x: x['jaccard'], reverse=True)
    #
    # merged_hyperedges_indices = set()
    #
    # # 这里的 hyperedge_sets 列表是可变的，我们会实时更新它
    # # 但为了逻辑清晰，我们标记索引
    #
    # JACCARD_THRESHOLD = 0.8
    # print("\n超边合并详情:")
    #
    # # 为了模拟 Julia 中直接修改 hyperedge_sets[keep_idx] 的行为，
    # # 我们需要能够索引到当前的集合状态。
    # # 由于 Python 的引用机制，直接修改列表中的 set 即可。
    #
    # for sim in hyperedge_similarities:
    #     if sim['jaccard'] < JACCARD_THRESHOLD:
    #         continue
    #
    #     i, j = sim['indices']
    #
    #     if i in merged_hyperedges_indices or j in merged_hyperedges_indices:
    #         continue
    #
    #     keep_idx = min(i, j)
    #     remove_idx = max(i, j)
    #
    #     print("\n合并超边:")
    #     print(f"  超边 {keep_idx} (节点: {sorted(list(hyperedge_sets[keep_idx]))})")
    #     print(f"  超边 {remove_idx} (节点: {sorted(list(hyperedge_sets[remove_idx]))})")
    #     print(f"  Jaccard相似度: {round(sim['jaccard'], 3)}")
    #
    #     # 合并：Union
    #     hyperedge_sets[keep_idx] = hyperedge_sets[keep_idx].union(hyperedge_sets[remove_idx])
    #     merged_hyperedges_indices.add(remove_idx)
    #
    #     print(f"  -> 新超边 {keep_idx} (节点: {sorted(list(hyperedge_sets[keep_idx]))})")
    #
    # # 构建最终超边列表
    # final_merged_hyperedges = []
    # for idx in range(n_he):
    #     if idx not in merged_hyperedges_indices:
    #         final_merged_hyperedges.append(sorted(list(hyperedge_sets[idx])))
    #
    # print(f"\n超边合并完成：原始超边数 = {n_he}，合并后超边数 = {len(final_merged_hyperedges)}")
    #
    # return final_nodes, final_merged_hyperedges
    # 只留下与 final_nodes 相关的超边

    # final_nodes_set = set(final_nodes)
    # filtered_hyperedges = []
    #
    # for hyperedge in final_hyperedges:
    #     if any(node in final_nodes_set for node in hyperedge):
    #         filtered_hyperedges.append(hyperedge)
    #
    # print(f"\n只留下与一跳节点相关的超边数量= {len(filtered_hyperedges)}")
    #
    # # === 修改在这里：直接返回结果，不再进行后续的 Jaccard 合并 ===
    # return final_nodes, filtered_hyperedges
    # 8. 使用超边稀疏化算法进行采样 (替代原有的 Jaccard 合并)

    # 8.1 准备输入数据：将列表转换为字典 {id: nodes}
    hyperedge_dict_input = {i: he for i, he in enumerate(final_hyperedges)}
    total_edges = len(hyperedge_dict_input)

    # 8.2 确定要删除的边数 k
    # 设定目标保留的超边数量 (TARGET_KEEP_NUM)
    # 你可以根据 Token 限制调整这个数字，例如 15 或 20
    k_remove = max(0, total_edges - TARGET_KEEP_NUM)

    final_processed_edges = []

    if k_remove > 0:
        print(f"\n[超边采样] 触发稀疏化: 当前 {total_edges} 条, 目标保留 {TARGET_KEEP_NUM} 条, 计划删除 {k_remove} 条")

        # 8.3 调用快速稀疏化函数
        # 注意：tune_nodes_num 设为较小的值(如 30)可以加快速度
        sampled_dict = hyperedge_sparsification_incremental_fast(
            hyperedge_dict=hyperedge_dict_input,
            k=k_remove,
            tune_nodes_num=min(30, len(final_nodes)),
            random_state=random.Random(42)  # 固定种子保证结果可复现
        )

        # 8.4 将结果转换回列表
        final_processed_edges = list(sampled_dict.values())
    else:
        print(f"\n[超边采样] 数量 ({total_edges}) 未超过阈值 ({TARGET_KEEP_NUM})，跳过采样。")
        final_processed_edges = final_hyperedges

    # 8.5 最终的安全过滤 (可选，确保采样后没有空边)
    final_clean_edges = [sorted(he) for he in final_processed_edges if len(he) > 0]

    print(f"最终返回超边数量: {len(final_clean_edges)}")

    # 直接返回结果
    return final_nodes, final_clean_edges


def coarse_hypergraph_int_4_2hop_MC(
        node_1hop: list[int],
        hyperedges_1hop_dict: dict[int, list[int]],
        hyperedges_2hop_dict: dict[int, list[int]],
        K:int,
        current_community: list[int] | None = None,
) -> tuple[list[int], defaultdict[Any, list], dict[str, float] | dict[str, float | Any]]:
    # 增加模块度筛选，选择提升社区模块度最大的前K个节点
    # 1. 创建本地副本
    local_hyperedges_1hop = copy.deepcopy(hyperedges_1hop_dict)
    local_hyperedges_2hop = copy.deepcopy(hyperedges_2hop_dict)
    node_1hop = sorted(node_1hop)
    # 第一阶段还是进行结构相似节点的合并
    # 2. 计算节点度（固定顺序遍历）
    degree_dict = {node: 0 for node in node_1hop}

    # 按 Key 排序遍历
    for hyperedge in local_hyperedges_1hop.values():
        # 不确定超边 list 内部是否有重复节点，可以加上 set() 去重
        for node in set(hyperedge): 
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
    SIMILARITY_THRESHOLD = 0.9
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