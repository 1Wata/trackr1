import json
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, DatasetDict
import re


def image_to_bytes(image_path: str) -> bytes:
    """
    从图像路径加载图像，并将其转换为字节数据。
    """
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        # 尝试使用 Pillow 加载图像
        with Image.open(image_path) as image:
            image = image.convert("RGB")  # 强制转换为 RGB 模式
            with BytesIO() as buffer:
                image.save(buffer, format='JPEG')  # 保存为 JPEG 格式
                return buffer.getvalue()
    except Exception as e:
        print(f"Pillow failed to load image from {image_path}: {e}")
        print("Trying OpenCV...")
        
        # 尝试使用 OpenCV 加载图像
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            pil_image = Image.fromarray(image_rgb)
            with BytesIO() as buffer:
                pil_image.save(buffer, format='JPEG')  # 保存为 JPEG 格式
                return buffer.getvalue()
        else:
            print(f"OpenCV failed to load image from {image_path}.")
    return None

def process_row(row: dict, image_key: str) -> dict:
    """
    处理单条数据，将 `images` 字段中的图像路径加载为字节数据。
    """
    # row.pop('id',None)
    if image_key in row:
        images = row[image_key]
        if isinstance(images, list):  # 如果是列表，逐个处理
            row[image_key] = [
                {"bytes": image_to_bytes(image_path)} for image_path in images if os.path.exists(image_path)
            ]
        elif isinstance(images, str):  # 单个图像路径
            if os.path.exists(images):  # 检查路径是否存在
                row[image_key] = {"bytes": image_to_bytes(images)}
            else:
                row[image_key] = None  # 如果路径不存在，设置为 None
    return row

# 新的辅助函数，用于转换单条记录
def transform_row(original_row: dict, input_image_key: str, output_image_key_name: str) -> dict:
    """
    转换单条记录以匹配目标 Parquet 结构:
    1. 如果原始数据中存在 'solution' 键，则将其内容视为 'answer' 进行处理，并在输出中统一使用 'answer' 键。
       如果 'solution' 不存在但 'answer' 存在，则使用 'answer'。
    2. 'answer' 字段 (源自 'solution' 或 'answer') 如果包含 <answer>...</answer> 标签，则移除标签保留内部内容。
       如果移除标签后的内容是形如 "[x,y,w,h]" 的字符串，则进一步转换为 "x,y,w,h"。
    3. 从原始 'prompt' 字段的 'content' 列表中提取文本，合并为新的 'problem' 字段。原始的 'problem' 字段将被丢弃。
    4. 将 input_image_key 中的图像路径转换为字节数据，并存入 output_image_key_name。
    5. 最终输出的字典将只包含 'problem', 'answer', 和 output_image_key_name (例如 'images') 这几个键。
    """
    final_row = {}

    # 1. 处理 'solution' 或 'answer' 字段
    raw_answer_content = None
    if 'solution' in original_row:
        raw_answer_content = original_row['solution']
    elif 'answer' in original_row: # 仅当 'solution' 不存在时
        raw_answer_content = original_row['answer']

    processed_answer = ""
    if isinstance(raw_answer_content, str):
        match = re.match(r"<answer>(.*)</answer>", raw_answer_content, re.DOTALL)
        if match:
            processed_answer = match.group(1).strip()
        else:
            processed_answer = raw_answer_content.strip()
        
        # 新增逻辑：如果 answer 是 "[x,y,w,h]" 格式的字符串，转换为 "x,y,w,h"
        if processed_answer.startswith('[') and processed_answer.endswith(']'):
            # 移除首尾的中括号
            content_inside_brackets = processed_answer[1:-1]
            # 按逗号分割，去除每个元素的空白，然后重新用逗号连接
            coords = [c.strip() for c in content_inside_brackets.split(',')]
            processed_answer = ",".join(coords)
            
    final_row['answer'] = processed_answer

    # 2. 从 'prompt' 字段的 'content' 中提取文本作为新的 'problem'
    # 原始的 'problem' 字段将被忽略
    problem_text_parts = []
    prompt_data = original_row.get('prompt')
    if isinstance(prompt_data, list) and len(prompt_data) > 0:
        # 假设我们关心的是 prompt 列表中的第一个元素的 content
        first_prompt_item = prompt_data[0]
        if isinstance(first_prompt_item, dict):
            content_list = first_prompt_item.get('content')
            if isinstance(content_list, list):
                for item in content_list:
                    # 提取 'text' 键的值，如果存在且不为 None
                    if isinstance(item, dict) and 'text' in item and item['text'] is not None:
                        problem_text_parts.append(str(item['text']).strip())
    
    final_row['problem'] = "\n".join(problem_text_parts) # 将所有文本部分用换行符连接

    # 3. 处理图像数据
    image_byte_list = []
    image_paths_source = original_row.get(input_image_key) 

    if image_paths_source:
        paths_to_process = []
        if isinstance(image_paths_source, list):
            # 确保列表中的路径是字符串
            paths_to_process = [p for p in image_paths_source if isinstance(p, str)]
        elif isinstance(image_paths_source, str): # 单个图像路径
            paths_to_process = [image_paths_source]
        
        for path in paths_to_process:
            img_bytes = image_to_bytes(path) # image_to_bytes 内部处理路径不存在的情况
            if img_bytes:
                image_byte_list.append({"bytes": img_bytes})
                
    final_row[output_image_key_name] = image_byte_list
        
    return final_row

def convert_json_to_parquet_with_datasets(json_path: str, parquet_path: str, image_key: str = "image", max_workers: int = 8):
    """
    使用 datasets 库将 JSON 文件转换为 Parquet 文件。
    此版本使用 transform_row 函数进行数据转换：
    - 'solution' 键重命名为 'answer'。
    - 'answer' 字段移除 <answer>...</answer> 标签。
    - 'prompt' 键被移除。
    - 图像路径（由 image_key 指定）被转换为字节数据并存储在 'images' 字段中。
    - 输出将只包含 'problem', 'answer', 'images' 字段。
    使用多线程加速图像处理。

    Args:
        json_path (str): 输入的 JSON 文件路径。
        parquet_path (str): 输出的 Parquet 文件路径。
        image_key (str): JSON 中表示图像路径的字段名 (例如 "image_files", "raw_images")。
                         默认为 "image"。这个字段的值会被 transform_row 用来加载图像字节。
        max_workers (int): 最大线程数，默认为 8。
    """
    # 读取 JSON 文件
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from JSON.")

    processed_data = []
    # 创建一个线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池，使用 transform_row 进行转换
        # output_image_key_name 固定为 "images" 以匹配目标 Parquet 结构
        futures = {executor.submit(transform_row, row, image_key, "images"): row for row in data}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                processed_row = future.result()
                if processed_row: 
                    processed_data.append(processed_row)
            except Exception as e:
                print(f"Error processing a row: {e}") # 记录单行处理错误
    
    if not processed_data:
        print("No data was processed successfully. Parquet file will not be generated.")
        return

    # 获取所有最终确定的键，以处理可能的空列表情况
    # transform_row 应该确保所有字典具有一致的键：'problem', 'answer', 'images'
    # 因此可以直接使用第一个元素的键，或预定义它们
    final_keys = ['problem', 'answer', 'images'] # 与 transform_row 输出保持一致
    
    columns = {key: [row.get(key) for row in processed_data] for key in final_keys}
    
    # 使用 datasets 创建数据集
    try:
        dataset = Dataset.from_dict(columns)
        # 保存为 Parquet 文件
        dataset.to_parquet(parquet_path)
        print(f"Parquet file saved to {parquet_path}")
    except Exception as e:
        print(f"Error creating Dataset or saving to Parquet: {e}")

if __name__ == '__main__':
    # 示例：假设你的JSON文件路径和输出Parquet文件路径
    json_file_path = "/data1/lihaobo/tracking/test/tracking_dataset.json" # 替换为你的JSON文件路径
    parquet_file_path = "/data1/lihaobo/track_r1/test.parquet" # 替换为你的输出Parquet文件路径
    
    # 调用转换函数，使用默认的 image_key="image"
    convert_json_to_parquet_with_datasets(json_file_path, parquet_file_path)
    import datasets
    # 检查生成的 Parquet 文件
    parquet_dataset = datasets.load_dataset("parquet", data_files=parquet_file_path)
    print(parquet_dataset['train'][1])  # 打印第一条记录以验证转换结果
    # 如果你的图像键名不是 "image"，例如是 "image_files"，则这样调用：
    # convert_json_to_parquet_with_datasets(json_file_path, parquet_file_path, image_key="image_files")