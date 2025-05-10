import re
from typing import List, Optional, Dict, Tuple

def parse_bbox_from_answer(answer_text: str) -> Optional[List[int]]:
    """
    从类似 "<answer>[x1, y1, x2, y2]</answer>" 的字符串中解析边界框。
    坐标应为整数。
    如果解析成功，返回 [x1, y1, x2, y2]，否则返回 None。
    """
    match = re.search(r"<answer>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</answer>", answer_text)
    if match:
        try:
            return [int(c) for c in match.groups()]
        except ValueError:
            return None
    return None

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)。
    每个框的格式为 [x1, y1, x2, y2]。
    坐标假定在 [0, 1000] 范围内。
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 确保两个框的 x1 < x2 且 y1 < y2
    if x1_1 >= x2_1 or y1_1 >= y2_1 or x1_2 >= x2_2 or y1_2 >= y2_2:
        return 0.0 # 无效框

    # 计算交集坐标
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # 计算交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    if intersection_area == 0:
        return 0.0

    # 计算各个框的面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0: # 如果 intersection_area > 0，则不应发生
        return 0.0

    iou = intersection_area / union_area
    return iou

def format_reward(predict: str) -> float:
    """
    检查预测字符串是否符合预期的 <answer>[x1,y1,x2,y2]</answer> 格式，
    并且包含有效的整数坐标。
    """
    parsed_bbox = parse_bbox_from_answer(predict)
    return 1.0 if parsed_bbox is not None else 0.0

def iou_accuracy_reward(predict: str, ground_truth_bbox_str: str) -> float:
    """
    根据预测和真实的边界框计算 IoU 得分。
    'predict' 是模型的完整输出字符串。
    'ground_truth_bbox_str' 是一个类似 "x1,y1,x2,y2" 的字符串。
    """
    pred_bbox = parse_bbox_from_answer(predict)
    if pred_bbox is None:
        return 0.0

    try:
        gt_coords = [int(c.strip()) for c in ground_truth_bbox_str.split(',')]
        if len(gt_coords) != 4:
            print(f"警告: 真实边界框字符串 '{ground_truth_bbox_str}' 不是 'x1,y1,x2,y2' 格式。")
            return 0.0
    except ValueError:
        print(f"警告: 无法解析真实边界框字符串 '{ground_truth_bbox_str}'。")
        return 0.0

    return calculate_iou(pred_bbox, gt_coords)

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    为一批预测和真实标签计算得分。
    'predicts': 模型输出字符串列表。
    'ground_truths': 真实边界框字符串列表，例如 ["100,150,300,350", ...]。
    'format_weight': 格式奖励在总得分中的权重。
    """
    scores = []
    if len(predicts) != len(ground_truths):
        print(f"错误: predicts ({len(predicts)}) 和 ground_truths ({len(ground_truths)}) 的长度不匹配。")
        min_len = min(len(predicts), len(ground_truths))
        predicts = predicts[:min_len]
        ground_truths = ground_truths[:min_len]

    for predict, ground_truth_str in zip(predicts, ground_truths):
        fmt_score = format_reward(predict)
        acc_score = iou_accuracy_reward(predict, ground_truth_str) # 这是 IoU 得分

        overall_score = (1 - format_weight) * acc_score + format_weight * fmt_score
        scores.append(
            {
                "overall": overall_score,
                "format": fmt_score,
                "iou": acc_score, # 使用 "iou" 作为键以保持清晰
            }
        )
    return scores