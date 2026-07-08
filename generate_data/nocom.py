import json
import re
import os


def extract_final_answer(cot_text):
    """
    从带有 CoT 的文本中提取最终答案。
    """
    if "Decision:" in cot_text:
        return cot_text.split("Decision:")[-1].strip()
    elif "Final Answer:" in cot_text:
        return cot_text.split("Final Answer:")[-1].strip()
    else:
        lines = [line.strip() for line in cot_text.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        return cot_text.strip()


def parse_community_nodes(community_str):
    """
    解析当前社区节点字符串（例如 "[23]" 或 "[23, 45]"）为整数集合。
    """
    try:
        cleaned_str = community_str.strip()
        if not cleaned_str.startswith('['):
            cleaned_str = f"[{cleaned_str}]"
        nodes = json.loads(cleaned_str)
        return set(int(x) for x in nodes)
    except Exception:
        nodes = re.findall(r'\d+', community_str)
        return set(int(x) for x in nodes)


def get_1_hop_hypergraph(current_community_set, global_hyperedges):
    """
    根据当前社区节点，从全局超边集中筛选出一跳局部超图。
    """
    local_hyperedges = []
    for edge in global_hyperedges:
        if current_community_set.intersection(edge):
            local_hyperedges.append(edge)
    return local_hyperedges


def process_dataset(input_path, output_path, global_hyperedges):
    print(f"开始读取训练数据集文件: {input_path}")

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]

    print(f"成功加载 {len(data)} 条训练数据。开始同步清洗并以 N-Set 语言构造局部超图...")

    cleaned_data = []
    success_count = 0

    for i, item in enumerate(data):
        old_instruction = item.get('instruction', '')
        old_output = str(item.get('output', ''))

        input_field = item.get('input', {})
        if not isinstance(input_field, dict):
            continue

        # ================= 功能 1: 清洗 INPUT (移除 community_state) =================
        if 'community_state' in input_field:
            input_field.pop('community_state')

        # ================= 功能 2: 构造并使用 N-Set 语言编码一跳局部超图 =================
        current_community_str = input_field.get('current_community', '[]')
        current_community_set = parse_community_nodes(current_community_str)

        # 抽取一跳局部超图
        local_hyperedges = get_1_hop_hypergraph(current_community_set, global_hyperedges)

        # 【核心修改点】：遵循论文 N-Set 编码规范：(v1, v2, v3)
        edge_list_str = []
        for value in local_hyperedges:
            # 将节点转换为带有 'v' 前缀的字符串，并用逗号+空格分隔，两端包裹圆括号
            nodes_str = ", ".join(f"v{node}" for node in value)
            edge_str = f"({nodes_str})"
            edge_list_str.append(edge_str)

        # 按照论文标准，所有超边通过逗号和空格连接成一整行描述
        # 如果你希望保持一行一条边，可以改为 "\n".join(edge_list_str)
        input_field['local_hypergraph'] = ", ".join(edge_list_str)
        item['input'] = input_field

        # ================= 功能 3: 清洗 OUTPUT (移除 COT 思维链) =================
        new_output = extract_final_answer(old_output)
        item['output'] = new_output

        # ================= 功能 4: 强化 INSTRUCTION (修改提示词) =================
        append_text = " Please directly output the final decision. Do not provide any reasoning or explanation."
        if "Do not provide any reasoning" not in old_instruction:
            item['instruction'] = old_instruction.strip() + append_text

        cleaned_data.append(item)

        # --- DEBUG 打印第一条数据转换效果 ---
        if i == 0:
            print("\n" + "=" * 60)
            print("【DEBUG: 第一条数据转换效果对比（已对齐论文 N-Set 编码）】")
            print(f"--- 当前社区节点解析结果 ---: {current_community_set}")
            print(f"--- 抽取到的一跳超图边数 ---: {len(local_hyperedges)}")
            print(f"--- 新 input 中的 local_hypergraph 编码结果 ---:")
            print(input_field['local_hypergraph'])
            print("-" * 40)
            print(f"--- 新 output 内容 ---: {item['output']}")
            print("=" * 60 + "\n")

        success_count += 1

    # ================= 以标准的 JSONL 格式保存 =================
    print(f"处理完成，共处理 {success_count} 条数据。正在保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("JSONL 文件保存成功！")


if __name__ == "__main__":
    # 基础路径配置
    BASE_PATH = r"D:\InstructCom\datasets\NTU2012"  # 请根据实际情况修改路径
    HYPEREDGE_FILE = os.path.join(BASE_PATH, "hyperedge.txt")
    INPUT_DATA_FILE = os.path.join(BASE_PATH, "525.json")
    OUTPUT_DATA_FILE = os.path.join(BASE_PATH, "525_nocom.json")

    # 1. 预先读入全局超边集
    global_hyperedges = []
    if not os.path.exists(HYPEREDGE_FILE):
        print(f"错误: 找不到超边集文件 '{HYPEREDGE_FILE}'")
    elif not os.path.exists(INPUT_DATA_FILE):
        print(f"错误: 找不到输入训练数据文件 '{INPUT_DATA_FILE}'")
    else:
        print(f"正在加载全局超边集文件: {HYPEREDGE_FILE}")
        with open(HYPEREDGE_FILE, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    row = [int(x) for x in line.split(',')]
                    global_hyperedges.append(row)
        print(f"全局超边集加载完成，共 {len(global_hyperedges)} 条超边。")

        # 2. 开始处理数据集
        process_dataset(INPUT_DATA_FILE, OUTPUT_DATA_FILE, global_hyperedges)