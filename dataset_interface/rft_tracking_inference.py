import os
import re
import json
import argparse
import logging
# import time # Not used
import torch
# import numpy as np # Not Used
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm
# import matplotlib.pyplot as plt # Not Used
from transformers import AutoProcessor, GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
# sys.path.append( # Assuming evaluation.datasets is accessible or part of EasyR1 structure
#     os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rft/src/virft/src'))
# )

# If evaluation.datasets is part of a different structure, adjust path or ensure PYTHONPATH is set
# For now, assuming it's findable. If it's part of EasyR1, it might be:
# from EasyR1.evaluation.datasets import get_dataset, SequenceList
# Or if this script is intended to be run from within dataset_interface:
from evaluation.datasets import get_dataset, SequenceList


import functools
import multiprocessing as mp
from multiprocessing import Pool
import math
from typing import List, Optional, Tuple
from analysis_results import evaluate_direct
ImageFile.LOAD_TRUNCATED_IMAGES = True # Helpful for some datasets

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants from EasyR1/verl/utils/dataset.py (or similar if they were args)
# These will now come from args: smart_resize_min_pixels, smart_resize_max_pixels
# DEFAULT_FACTOR = 28 # No longer used
DEFAULT_MIN_PIXELS = 3136  # Example, should match training if possible (56*56)
DEFAULT_MAX_PIXELS = 409600 # Example, should match training if possible (320*320)


# --- Tracking System Prompt (remains the same) ---
def get_tracking_system_prompt(use_thinking=False):
    """获取追踪任务的系统提示"""
    base_prompt = (
        "You are a professional visual object tracking assistant. Your task is to track a specified target object. "
        "The user will provide template frames showing the target object. "
        "First, identify the target in the template frames. Then, locate the target's position in the following search frames."
    )
    
    if use_thinking: 
        return base_prompt + (
            "You should first think about how to locate the target by analyzing visual features, then provide your answer. "
            "Put your thinking process inside <thinking>...</thinking> tags. "
            "Your final answer should be the target's bounding box coordinates in the format [x1, y1, x2, y2], "
            "where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
            "Wrap your final answer in <answer>[x1, y1, x2, y2]</answer> tags."
        )
    else:
        return base_prompt + (
            "Please directly return the target's bounding box coordinates in the format [x1, y1, x2, y2], "
            "where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
            "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
        )

TRACKING_SYSTEM_PROMPT = get_tracking_system_prompt(False)
# --- End ---

# --- Bbox utility functions (remain the same) ---
def convert_bbox_xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]

def convert_bbox_xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]



def extract_single_bbox(response):
    """
    从模型响应中提取单个边界框，自动适应多种响应格式
    
    Args:
        response (str): 模型的响应文本
        
    Returns:
        list: 提取的边界框坐标 [x1, y1, x2, y2]，如果无法提取则返回 None
    """
    # 尝试从不同格式中提取内容
    content_str = None
    
    # 检查是否有 thinking/answer 格式
    thinking_start = "<thinking>"
    thinking_end = "</thinking>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    
    # 如果存在 thinking 和 answer 标签
    if thinking_start in response and answer_start in response:
        # 提取 answer 部分
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果只有 answer 标签
    elif answer_start in response:
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果没有标签，则尝试直接提取
    else:
        content_str = response.strip()
    
    # 如果没有内容可提取
    if not content_str:
        return None
    
    # 替换单引号为双引号以兼容JSON格式
    content_str = content_str.replace("'", '"')
    
    # 尝试解析为JSON格式
    # 方法1: 直接的坐标列表 [x1, y1, x2, y2]
    if content_str.startswith('[') and content_str.endswith(']'):
        # 尝试解析为JSON数组
        import json
        
        # 尝试解析整个内容
        bbox_data = None
        try:
            bbox_data = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_data is not None:
            # 检查是直接的坐标列表还是带有Position键的对象列表
            if isinstance(bbox_data, list):
                if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                    # 直接返回坐标列表 [x1, y1, x2, y2]
                    return bbox_data
                elif bbox_data and isinstance(bbox_data[0], dict) and 'Position' in bbox_data[0]:
                    # 返回第一个边界框的Position
                    return bbox_data[0]['Position']
    
    # 方法2: 尝试解析为字典形式 {'Position': [x1, y1, x2, y2]}
    if content_str.startswith('{') and content_str.endswith('}'):
        import json
        
        bbox_dict = None
        try:
            bbox_dict = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_dict is not None and isinstance(bbox_dict, dict) and 'Position' in bbox_dict:
            return bbox_dict['Position']
    
    # 方法3: 使用正则表达式提取数字列表
    import re
    matches = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', content_str)
    if matches:
        return [int(x) for x in matches[0]]
    
    return None
# --- End Bbox Utils ---

# --- Model Loading (remains largely the same) ---
def load_model_and_processor(model_path, device="auto"):
    logger.info(f"Loading model from {model_path} to device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # The processor might also handle some image processing, but we'll do explicit resizing first
    processor = AutoProcessor.from_pretrained(model_path) 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None # Conditional Flash Attention
    )
    model.eval()
    return model, tokenizer, processor
# --- End Model Loading ---

# --- New Image Processing and Bbox Scaling Helpers (inspired by EasyR1/verl/utils/dataset.py) ---
def process_image_for_inference(
    image_pil: Image.Image, 
    min_pixels: int, 
    max_pixels: int
) -> Image.Image:
    """
    Processes a PIL image similar to ImageProcessMixin in EasyR1.
    Resizes based on min_pixels and max_pixels, and converts to RGB.
    """
    if not isinstance(image_pil, Image.Image):
        # This case should ideally be caught before calling this function
        logger.error("Invalid input: not a PIL Image object.")
        raise TypeError("Input must be a PIL Image object.")

    # Ensure min_pixels and max_pixels are valid
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        logger.warning(f"Invalid min_pixels ({min_pixels}) or max_pixels ({max_pixels}). Skipping resize based on pixel count.")
    else:
        current_pixels = image_pil.width * image_pil.height
        if current_pixels == 0: # Avoid division by zero for empty images
            logger.warning(f"Image has zero pixels (size: {image_pil.size}). Cannot resize.")
        else:
            if current_pixels > max_pixels:
                resize_factor = math.sqrt(max_pixels / current_pixels)
                width, height = int(image_pil.width * resize_factor), int(image_pil.height * resize_factor)
                if width > 0 and height > 0:
                     image_pil = image_pil.resize((width, height), Image.Resampling.BILINEAR)
                else:
                    logger.warning(f"Calculated zero dimension for resize based on max_pixels. Original: {image_pil.size}, Factor: {resize_factor}")
                current_pixels = image_pil.width * image_pil.height # Update current_pixels after resize

            if current_pixels > 0 and current_pixels < min_pixels : # Check current_pixels > 0 again
                resize_factor = math.sqrt(min_pixels / current_pixels)
                width, height = int(image_pil.width * resize_factor), int(image_pil.height * resize_factor)
                if width > 0 and height > 0:
                    image_pil = image_pil.resize((width, height), Image.Resampling.BILINEAR)
                else:
                    logger.warning(f"Calculated zero dimension for resize based on min_pixels. Original: {image_pil.size}, Factor: {resize_factor}")

    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    return image_pil

def scale_bbox_coordinates(
    bbox_xyxy: List[int], 
    scale_x: float, 
    scale_y: float,
    clamp_size: Optional[Tuple[int, int]] = None # (width, height) to clamp to
) -> Optional[List[int]]:
    """Scales bbox and optionally clamps it."""
    x1, y1, x2, y2 = bbox_xyxy

    x1_s = round(x1 * scale_x)
    y1_s = round(y1 * scale_y)
    x2_s = round(x2 * scale_x)
    y2_s = round(y2 * scale_y)

    if clamp_size and clamp_size[0] > 0 and clamp_size[1] > 0:
        clamp_w, clamp_h = clamp_size
        x1_s = max(0, min(x1_s, clamp_w - 1))
        y1_s = max(0, min(y1_s, clamp_h - 1))
        x2_s = max(0, min(x2_s, clamp_w - 1))
        y2_s = max(0, min(y2_s, clamp_h - 1))
    
    # Ensure x1 <= x2 and y1 <= y2 after scaling and clamping
    if x1_s > x2_s: x1_s, x2_s = x2_s, x1_s 
    if y1_s > y2_s: y1_s, y2_s = y2_s, y1_s

    if x1_s >= x2_s or y1_s >= y2_s: # Check for invalid box (width/height is zero or negative)
        # logger.warning(f"Bbox {bbox_xyxy} became invalid [{x1_s},{y1_s},{x2_s},{y2_s}] after scaling/clamping. Scale factors: ({scale_x}, {scale_y}), Clamp: {clamp_size}")
        return None 
        
    return [x1_s, y1_s, x2_s, y2_s]
# --- End New Helpers ---


# --- Prompt Building (expects resized images and bboxes in resized image coordinates) ---
def build_rft_input_messages(exp_str, template_pils_resized, search_pil_resized, template_bboxes_resized_for_text_prompt):
    user_content_list_of_dicts = []
    for _ in template_pils_resized: 
        user_content_list_of_dicts.append({"type": "image"})
    
    if template_pils_resized:
        for idx, template_bbox_in_resized_coords in enumerate(template_bboxes_resized_for_text_prompt):
            if template_bbox_in_resized_coords: 
                bbox_text_content = (
                    f"The bounding box for template frame {idx + 1} is: "
                    f"[{int(template_bbox_in_resized_coords[0])}, {int(template_bbox_in_resized_coords[1])}, "
                    f"{int(template_bbox_in_resized_coords[2])}, {int(template_bbox_in_resized_coords[3])}]."
                )
                user_content_list_of_dicts.append({"type": "text", "text": "\n" + bbox_text_content})
            else: 
                user_content_list_of_dicts.append({"type": "text", "text": f"\nTemplate frame {idx + 1} is provided without a valid bounding box for text prompt."})

    if template_pils_resized:
        object_description_text = f"\nThese are the template frames showing the object '{exp_str}'."
        user_content_list_of_dicts.append({"type": "text", "text": object_description_text})
    
    user_content_list_of_dicts.append({"type": "image"}) # For the search_pil_resized
    
    tracking_instruction_text = (
        f" Please track the object '{exp_str}' in the search frame provided after the template frames. "
        "Provide a bounding box for this search frame. "
        "The bounding box should be in [x1, y1, x2, y2] format, relative to this search frame's (potentially resized) dimensions. "
        "Wrap your answer in <answer>[x1, y1, x2, y2]</answer> tags."
    )
    user_content_list_of_dicts.append({"type": "text", "text": tracking_instruction_text})
    
    messages = [
        {"role": "system", "content": TRACKING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content_list_of_dicts}
    ]
    return messages
# --- End Prompt Building ---

# --- Visualization (remains the same) ---
def draw_bbox_on_image(image_pil, bbox_xyxy, color="red", width=2):
    if bbox_xyxy is None:
        return image_pil
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(bbox_xyxy, outline=color, width=width)
    return img_draw
# --- End Visualization ---

# --- History Management (stores ORIGINAL paths and ORIGINAL bboxes) ---
def update_history_indexed(
    history_frame_paths: List[Optional[str]], 
    history_bboxes_orig_xyxy: List[Optional[List[int]]], 
    current_frame_idx: int,
    current_frame_path: Optional[str], 
    current_bbox_orig_xyxy: Optional[List[int]] 
):
    while len(history_frame_paths) <= current_frame_idx:
        history_frame_paths.append(None)
    while len(history_bboxes_orig_xyxy) <= current_frame_idx:
        history_bboxes_orig_xyxy.append(None)
    
    history_frame_paths[current_frame_idx] = current_frame_path
    history_bboxes_orig_xyxy[current_frame_idx] = current_bbox_orig_xyxy 
    return history_frame_paths, history_bboxes_orig_xyxy

def get_template_frames_by_gap(
    current_frame_idx: int,
    gap_list: List[int],
    all_history_frame_paths: List[Optional[str]], 
    all_history_bboxes_orig_xyxy: List[Optional[List[int]]], 
    first_frame_path: str, 
    first_frame_bbox_orig_xyxy: List[int]  
) -> Tuple[List[Image.Image], List[List[int]]]: 
    selected_template_pils_orig = []
    selected_template_bboxes_orig_xyxy = [] 
    temp_selected_paths_bboxes = []

    for gap in gap_list:
        if gap <= 0: 
            logger.warning(f"Invalid gap value {gap} <= 0, skipping.")
            continue
        template_frame_idx = current_frame_idx - gap
        path_to_load = None
        bbox_to_use_orig = None 

        if template_frame_idx >= 0: 
            if template_frame_idx < len(all_history_frame_paths) and \
               all_history_frame_paths[template_frame_idx] is not None and \
               all_history_bboxes_orig_xyxy[template_frame_idx] is not None:
                path_to_load = all_history_frame_paths[template_frame_idx]
                bbox_to_use_orig = all_history_bboxes_orig_xyxy[template_frame_idx]
            else: 
                logger.warning(f"History missing or invalid for frame index {template_frame_idx} (current_frame_idx: {current_frame_idx}, gap: {gap}). Using first frame as fallback.")
                path_to_load = first_frame_path
                bbox_to_use_orig = first_frame_bbox_orig_xyxy
        else: 
            path_to_load = first_frame_path
            bbox_to_use_orig = first_frame_bbox_orig_xyxy
        
        if path_to_load and bbox_to_use_orig:
            is_duplicate_path = any(p == path_to_load for p, _ in temp_selected_paths_bboxes)
            if not is_duplicate_path:
                temp_selected_paths_bboxes.append((path_to_load, bbox_to_use_orig))

    for path, bbox_orig in temp_selected_paths_bboxes:
        try:
            pil_image = Image.open(path).convert("RGB")
            selected_template_pils_orig.append(pil_image)
            selected_template_bboxes_orig_xyxy.append(bbox_orig) 
        except FileNotFoundError:
            logger.error(f"Template frame not found: {path}. Skipping.")
        except Exception as e: # Catch other PIL errors
            logger.error(f"Error loading template frame {path}: {e}. Skipping.")

    if not selected_template_pils_orig and current_frame_idx >= 0 : 
        logger.info("No templates selected via gap_list or loading failed, defaulting to first frame as the sole template.")
        try:
            pil_image = Image.open(first_frame_path).convert("RGB")
            selected_template_pils_orig.append(pil_image)
            selected_template_bboxes_orig_xyxy.append(first_frame_bbox_orig_xyxy)
        except Exception as e:
            logger.error(f"Failed to load default first frame template {first_frame_path}: {e}")
            
    return selected_template_pils_orig, selected_template_bboxes_orig_xyxy
# --- End History Management ---


def evaluate_tracking_rft(model, tokenizer, processor, sequences_names=None, dataset_name='lasot',
                          save_visualize=False, output_dir="rft_tracking_results", 
                          max_new_tokens=100, rank=0, 
                          smart_resize_min_pixels=DEFAULT_MIN_PIXELS, 
                          smart_resize_max_pixels=DEFAULT_MAX_PIXELS,
                          gap_list: Optional[List[int]] = None):
    
    if gap_list is None:
        gap_list = [1, 5] 
    logger.info(f"Process {rank} - Using gap_list: {gap_list}, min_pixels: {smart_resize_min_pixels}, max_pixels: {smart_resize_max_pixels}")

    full_dataset = get_dataset(dataset_name)
    results_summary = []

    sequences_to_process_for_this_rank = []
    if sequences_names: # sequences_names is the list of sequence names for this rank
        # Create a map of all sequence names to their objects from the full_dataset for efficient lookup
        all_sequences_map = {seq.name: seq for seq in full_dataset}
        for name in sequences_names:
            if name in all_sequences_map:
                sequences_to_process_for_this_rank.append(all_sequences_map[name])
            else:
                logger.warning(f"Process {rank} - Sequence name '{name}' assigned but not found in dataset '{dataset_name}'. Skipping.")
        
        if not sequences_to_process_for_this_rank:
            logger.info(f"Process {rank} - No valid sequences to process after filtering by names: {sequences_names}.")
            return [] # Return empty list if no sequences are left for this rank
    elif rank == 0 and not sequences_names: # Likely single process mode, or args.sequence was None
        sequences_to_process_for_this_rank = list(full_dataset) # Process all if no specific names given (typical for single process)
        if not sequences_to_process_for_this_rank and dataset_name:
             logger.info(f"Process {rank} - Dataset '{dataset_name}' is empty or no sequences were provided. Processing 0 sequences.")
             return []
    else: # sequences_names is empty or None, and not rank 0 in a way that implies "all"
        logger.info(f"Process {rank} - No sequences assigned or dataset is empty. Processing 0 sequences.")
        return []

    logger.info(f"Process {rank} - Will process {len(sequences_to_process_for_this_rank)} sequences: {[s.name for s in sequences_to_process_for_this_rank]}")

    # The main loop should iterate over 'sequences_to_process_for_this_rank'
    for seq_idx, seq in enumerate(tqdm(sequences_to_process_for_this_rank, desc=f"Process {rank} - Tracking progress")):
        # OLD path construction:
        # seq_output_dir = os.path.join(output_dir, dataset_name, seq.name)
        # os.makedirs(seq_output_dir, exist_ok=True)
        # predictions_file_path = os.path.join(seq_output_dir, "predictions.txt")

        # NEW path construction for analysis script compatibility:
        dataset_results_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)
        predictions_file_path = os.path.join(dataset_results_dir, f"{seq.name}.txt")
        
        # Visualization saving path needs adjustment if you want to keep them.
        # Option: Save visualizations in a separate structure or a subfolder.
        # For example, a subfolder within dataset_results_dir named after the sequence for visualizations:
        vis_output_dir_for_seq = os.path.join(dataset_results_dir, seq.name + "_visuals")
        if save_visualize:
            os.makedirs(vis_output_dir_for_seq, exist_ok=True)

        all_history_frame_paths: List[Optional[str]] = [] 
        all_history_bboxes_orig_xyxy: List[Optional[List[int]]] = [] 

        first_frame_path = seq.frames[0]
        try:
            first_frame_pil_orig = Image.open(first_frame_path).convert("RGB")
        except Exception as e:
            logger.error(f"Sequence {seq.name}: Failed to load first frame {first_frame_path}: {e}. Skipping sequence.")
            continue
        original_size_seq = first_frame_pil_orig.size 

        processed_first_frame_pil = process_image_for_inference(
            first_frame_pil_orig.copy(), smart_resize_min_pixels, smart_resize_max_pixels
        )
        resized_size_seq = processed_first_frame_pil.size
        
        scale_x_seq, scale_y_seq = 1.0, 1.0
        if original_size_seq[0] > 0 and original_size_seq[1] > 0:
            scale_x_seq = resized_size_seq[0] / original_size_seq[0]
            scale_y_seq = resized_size_seq[1] / original_size_seq[1]
        else:
            logger.error(f"Sequence {seq.name}: Invalid original_size_seq {original_size_seq}. Cannot calculate scales. Skipping sequence.")
            continue 

        init_info = seq.init_info()
        first_frame_gt_bbox_xywh = init_info.get('init_bbox')
        if not first_frame_gt_bbox_xywh or len(first_frame_gt_bbox_xywh) != 4:
            logger.error(f"Sequence {seq.name}: Invalid or missing init_bbox. Skipping sequence.")
            continue
        first_frame_gt_bbox_original_xyxy = convert_bbox_xywh_to_xyxy(first_frame_gt_bbox_xywh)
        exp_str = init_info.get('init_text_description', "the target object")

        first_frame_bbox_resized_for_prompt = scale_bbox_coordinates(
            first_frame_gt_bbox_original_xyxy, scale_x_seq, scale_y_seq, resized_size_seq
        )

        if save_visualize: 
            img_draw = draw_bbox_on_image(first_frame_pil_orig, first_frame_gt_bbox_original_xyxy)
            # OLD: img_draw.save(os.path.join(seq_output_dir, f"frame_0000_gt.jpg"))
            img_draw.save(os.path.join(vis_output_dir_for_seq, f"frame_0000_gt.jpg")) # NEW if using vis_output_dir_for_seq

        with open(predictions_file_path, "w") as f:
            f.write(f"{first_frame_gt_bbox_xywh[0]},{first_frame_gt_bbox_xywh[1]},{first_frame_gt_bbox_xywh[2]},{first_frame_gt_bbox_xywh[3]}\n")
        
        all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
            all_history_frame_paths, all_history_bboxes_orig_xyxy,
            0, first_frame_path, list(first_frame_gt_bbox_original_xyxy)
        )

        for i in range(1, len(seq.frames)):
            current_prompt_template_pils_resized_for_model = [] 
            current_prompt_template_bboxes_resized_for_text = []
            
            selected_template_pils_orig, selected_template_bboxes_orig_xyxy = get_template_frames_by_gap(
                current_frame_idx=i, gap_list=gap_list,
                all_history_frame_paths=all_history_frame_paths,
                all_history_bboxes_orig_xyxy=all_history_bboxes_orig_xyxy,
                first_frame_path=first_frame_path, 
                first_frame_bbox_orig_xyxy=first_frame_gt_bbox_original_xyxy
            )
            
            if not selected_template_pils_orig: 
                 logger.warning(f"No template frames selected by gap for frame {i}. Using first frame as sole prompt template.")
                 template_pil_resized = process_image_for_inference(first_frame_pil_orig.copy(), smart_resize_min_pixels, smart_resize_max_pixels)
                 current_prompt_template_pils_resized_for_model.append(template_pil_resized)
                 current_prompt_template_bboxes_resized_for_text.append(first_frame_bbox_resized_for_prompt) 
            else:
                for template_pil_orig, template_bbox_orig_xyxy in zip(selected_template_pils_orig, selected_template_bboxes_orig_xyxy):
                    if template_pil_orig is None or template_bbox_orig_xyxy is None: 
                        continue
                    template_pil_resized = process_image_for_inference(template_pil_orig.copy(), smart_resize_min_pixels, smart_resize_max_pixels)
                    
                    bbox_resized_for_text = scale_bbox_coordinates(
                        template_bbox_orig_xyxy, scale_x_seq, scale_y_seq, template_pil_resized.size # Clamp to specific template's resized size
                    )
                    if bbox_resized_for_text: 
                        current_prompt_template_pils_resized_for_model.append(template_pil_resized)
                        current_prompt_template_bboxes_resized_for_text.append(bbox_resized_for_text)
            
            if not current_prompt_template_pils_resized_for_model: 
                logger.error(f"CRITICAL: No valid prompt templates for frame {i} of sequence {seq.name}. Skipping frame.")
                with open(predictions_file_path, "a") as f: f.write("0,0,0,0\n") 
                all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
                    all_history_frame_paths, all_history_bboxes_orig_xyxy, i, seq.frames[i], None 
                )
                continue

            current_search_frame_path = seq.frames[i]
            try:
                search_pil_orig = Image.open(current_search_frame_path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load search frame {current_search_frame_path} for frame {i} of sequence {seq.name}: {e}. Skipping frame.")
                with open(predictions_file_path, "a") as f: f.write("0,0,0,0\n")
                all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
                    all_history_frame_paths, all_history_bboxes_orig_xyxy, i, current_search_frame_path, None
                )
                continue
            
            search_pil_resized_for_model = process_image_for_inference(search_pil_orig.copy(), smart_resize_min_pixels, smart_resize_max_pixels)

            messages = build_rft_input_messages(
                exp_str, 
                current_prompt_template_pils_resized_for_model, 
                search_pil_resized_for_model, 
                current_prompt_template_bboxes_resized_for_text
            )
            text_prompt_with_placeholders = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_images_for_sample = current_prompt_template_pils_resized_for_model + [search_pil_resized_for_model]
            
            inputs = processor(
                text=[text_prompt_with_placeholders], images=all_images_for_sample,
                return_tensors="pt", padding=True,
            ).to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, generation_config=gen_config)
            
            input_token_len = inputs["input_ids"].shape[1]
            generated_text_ids = generated_ids[:, input_token_len:]
            response_text = processor.batch_decode(generated_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            predicted_bbox_resized_xyxy = extract_single_bbox(response_text)
            
            pred_bbox_xywh_final_for_file = [0,0,0,0] 
            pred_bbox_original_xyxy_for_history = None

            if predicted_bbox_resized_xyxy:
                pred_bbox_original_xyxy_for_history = scale_bbox_coordinates(
                    predicted_bbox_resized_xyxy, 
                    1.0/scale_x_seq if scale_x_seq != 0 else 1.0, 
                    1.0/scale_y_seq if scale_y_seq != 0 else 1.0, 
                    search_pil_orig.size # Clamp to original search image size
                )
                if pred_bbox_original_xyxy_for_history:
                    pred_bbox_xywh_final_for_file = convert_bbox_xyxy_to_xywh(pred_bbox_original_xyxy_for_history)
            
            with open(predictions_file_path, "a") as f:
                f.write(f"{pred_bbox_xywh_final_for_file[0]},{pred_bbox_xywh_final_for_file[1]},{pred_bbox_xywh_final_for_file[2]},{pred_bbox_xywh_final_for_file[3]}\n")

            if save_visualize: 
                img_draw = draw_bbox_on_image(search_pil_orig, pred_bbox_original_xyxy_for_history)
                # OLD: img_draw.save(os.path.join(seq_output_dir, f"frame_{i:04d}_pred.jpg"))
                img_draw.save(os.path.join(vis_output_dir_for_seq, f"frame_{i:04d}_pred.jpg")) # NEW if using vis_output_dir_for_seq

            all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
                all_history_frame_paths, all_history_bboxes_orig_xyxy,
                i, current_search_frame_path, pred_bbox_original_xyxy_for_history 
            )

        results_summary.append({'sequence': seq.name, 'output_file': predictions_file_path})
        logger.info(f"Process {rank} - Finished sequence {seq.name}. Predictions saved to {predictions_file_path}")
        
    return results_summary

def split_sequences(seq_list_all_names, rank, world_size):
    if not seq_list_all_names: return []
    n_total = len(seq_list_all_names)
    n_per_rank = math.ceil(n_total / world_size)
    start_idx = rank * n_per_rank
    end_idx = min((rank + 1) * n_per_rank, n_total)
    return seq_list_all_names[start_idx:end_idx] if start_idx < n_total else []

def run_process_wrapper(rank, world_size, args, current_dataset_name_for_wrapper):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(rank)
    
    logger.info(f"Process {rank} started on device {device} for dataset {current_dataset_name_for_wrapper}")
    model, tokenizer, processor = load_model_and_processor(args.model_path, device=device)
    
    all_sequence_names_for_current_dataset = [seq.name for seq in get_dataset(current_dataset_name_for_wrapper)]

    if args.sequence:
        if rank == 0: # Only rank 0 handles the specific sequence if provided
            sequences_for_this_rank = [args.sequence]
            # Check if the sequence exists in the current dataset
            if args.sequence not in all_sequence_names_for_current_dataset:
                logger.warning(f"Process {rank} - Sequence '{args.sequence}' not found in dataset '{current_dataset_name_for_wrapper}'. This rank will process no sequences for this dataset.")
                sequences_for_this_rank = []
        else:
            sequences_for_this_rank = [] # Other ranks do nothing if a specific sequence is requested
    else:
        sequences_for_this_rank = split_sequences(all_sequence_names_for_current_dataset, rank, world_size)

    if not sequences_for_this_rank:
        logger.info(f"Process {rank} has no sequences to process for dataset {current_dataset_name_for_wrapper}.")
        return []

    logger.info(f"Process {rank} will process sequences: {sequences_for_this_rank} for dataset {current_dataset_name_for_wrapper}")
    
    return evaluate_tracking_rft(
        model, tokenizer, processor,
        sequences_names=sequences_for_this_rank,
        dataset_name=current_dataset_name_for_wrapper,
        save_visualize=args.save_vis,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        rank=rank,
        smart_resize_min_pixels=args.smart_resize_min_pixels,
        smart_resize_max_pixels=args.smart_resize_max_pixels,
        gap_list=args.gap_list
    )

def main():
    parser = argparse.ArgumentParser(description="RFT Model Single Object Tracking Inference (EasyR1 Style Processing)")
    parser.add_argument("--model_path", type=str, default='/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-3', help="Path to the fine-tuned RFT model")
    parser.add_argument("--dataset_name", type=str, nargs='+', default=["lasot", "TNL2k", "OTB_lang"], help="Dataset name(s), e.g., lasot TNL2k OTB_lang")
    parser.add_argument("--sequence", type=str, default=None, help="Specific sequence name (optional, will be attempted for each dataset)")
    parser.add_argument("--output_dir", type=str, default="rft_tracking_results_easyr1_style", help="Output directory")
    parser.add_argument("--save_vis", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save visualizations (true/false)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation")
    parser.add_argument("--smart_resize_min_pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Min pixels for image processing (like EasyR1)")
    parser.add_argument("--smart_resize_max_pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Max pixels for image processing (like EasyR1)")
    parser.add_argument("--gap_list", type=int, nargs='+', default=None, help="List of gaps for selecting template frames (e.g., --gap_list 1 5). If None, uses default [1, 10].")
    parser.add_argument("--single_process", action="store_true", help="Force single process")
    args = parser.parse_args()

    if args.gap_list is None:
        args.gap_list = [1, 10] 
    logger.info(f"Using gap_list: {args.gap_list}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    for current_dataset_name in args.dataset_name:
        logger.info(f"Processing dataset: {current_dataset_name}")
        
        use_multiprocessing = torch.cuda.device_count() > 1 and not args.single_process
        current_dataset_summary = []

        if use_multiprocessing:
            world_size = torch.cuda.device_count()
            logger.info(f"Detected {world_size} GPUs, enabling multi-process evaluation for dataset {current_dataset_name}.")
            if mp.get_start_method(allow_none=True) != 'spawn': 
                mp.set_start_method('spawn', force=True)
            
            # Pass current_dataset_name to the wrapper
            func = functools.partial(run_process_wrapper, world_size=world_size, args=args, current_dataset_name_for_wrapper=current_dataset_name)
            with Pool(world_size) as pool:
                results_list_of_lists = pool.map(func, range(world_size))
            current_dataset_summary = [item for sublist in results_list_of_lists for item in sublist if sublist] 
        else:
            logger.info(f"Using single process evaluation for dataset {current_dataset_name}.")
            rank = 0
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model, tokenizer, processor = load_model_and_processor(args.model_path, device=device)
            
            all_sequence_names_for_current_dataset = [seq.name for seq in get_dataset(current_dataset_name)]
            sequences_to_run = []
            if args.sequence:
                if args.sequence in all_sequence_names_for_current_dataset:
                    sequences_to_run = [args.sequence]
                else:
                    logger.warning(f"Sequence '{args.sequence}' not found in dataset '{current_dataset_name}'. Skipping this sequence for this dataset.")
            else:
                sequences_to_run = all_sequence_names_for_current_dataset

            if not sequences_to_run:
                logger.info(f"No sequences selected to run for dataset {current_dataset_name}.")
            else:
                current_dataset_summary = evaluate_tracking_rft(
                    model, tokenizer, processor, sequences_names=sequences_to_run,
                    dataset_name=current_dataset_name, save_visualize=args.save_vis,
                    output_dir=args.output_dir, max_new_tokens=args.max_new_tokens, rank=rank,
                    smart_resize_min_pixels=args.smart_resize_min_pixels,
                    smart_resize_max_pixels=args.smart_resize_max_pixels,
                    gap_list=args.gap_list
                )

        summary_file = os.path.join(args.output_dir, f"tracking_summary_{current_dataset_name}.json")
        with open(summary_file, "w") as f:
            json.dump(current_dataset_summary, f, indent=2)
        logger.info(f"Tracking summary for {current_dataset_name} saved to {summary_file}")

        logger.info(f"Proceeding to results analysis for dataset: {current_dataset_name}...")
        results_path_for_analysis = os.path.join(args.output_dir, current_dataset_name)
        
        logger.info(f"Calling evaluate_direct with results_path: {results_path_for_analysis} and dataset_name: {current_dataset_name}")
        
        evaluate_direct(results_path_for_analysis, current_dataset_name)


if __name__ == "__main__":
    main()