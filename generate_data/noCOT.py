import json
import re
import os

def extract_final_answer(cot_text):
    """
    从带有 CoT 的文本中提取最终答案。
    """
    # 策略 1: 寻找明确的 "Decision:" 分隔符
    if "Decision:" in cot_text:
        # 分割并取最后一部分，去除两端空白字符
        return cot_text.split("Decision:")[-1].strip()

    # 策略 2: 寻找可能存在的 "Final Answer:"
    elif "Final Answer:" in cot_text:
        return cot_text.split("Final Answer:")[-1].strip()
    
    # 策略 3 (兜底): 如果没有标志词，直接取最后一行非空文本
    else:
        lines = [line.strip() for line in cot_text.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        return cot_text.strip()


def process_dataset(input_path, output_path):
    print(f"开始读取文件: {input_path}")
    # 兼容加载 JSON (列表格式) 或 JSONL (按行格式)
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
    print(f"成功加载 {len(data)} 条数据。正在处理...")
    cleaned_data = []
    success_count = 0

    for i, item in enumerate(data):
        # 1. 提取旧的 output
        old_output = str(item.get('output', ''))
        # 2. 清洗提取纯净答案
        new_output = extract_final_answer(old_output)
        # 3. 更新 output 字段
        item['output'] = new_output
        # 4. (可选但推荐) 修改 instruction，强化“直接输出”的指令
        # 如果原指令里没有明确禁止推理，我们可以在末尾加上一句
        old_instruction = item.get('instruction', '')
        append_text = " Please directly output the final decision. Do not provide any reasoning or explanation."
        if "Do not provide any reasoning" not in old_instruction:
            item['instruction'] = old_instruction.strip() + append_text
        cleaned_data.append(item)
        # 打印第一条数据作为对照检查

        if i == 0:
            print("\n" + "=" * 50)
            print("【DEBUG: 第一条数据转换效果对比】")
            print(f"原 Output 长度: {len(old_output)} 字符")
            print(f"--- 原 Output 内容 ---\n{old_output}")
            print("\n" + "-" * 30)
            print(f"新 Output 长度: {len(new_output)} 字符")
            print(f"--- 新 Output 内容 ---\n{new_output}")
            print("=" * 50 + "\n")
        success_count += 1

    # 保存新数据集
    # 默认保存为普通 JSON 格式。如果你需要 JSONL，请改用 json.dumps 逐行写入
    print(f"处理完成，共清洗 {success_count} 条数据。正在保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
    print("保存成功！")


if __name__ == "__main__":

    # 请替换为你真实的旧数据集路径和想要保存的新数据集路径
    INPUT_FILE = r"D:\InstructCom\datasets\contact-high-school\612.json"
    OUTPUT_FILE = r"D:\InstructCom\datasets\contact-high-school\612_noCOT.json"

    # 如果文件不存在，给个友好提示避免报错
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'，请检查路径。")
    else:
        process_dataset(INPUT_FILE, OUTPUT_FILE)

