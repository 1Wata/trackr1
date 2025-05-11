import re
from typing import Dict
import json

def check_format(s):
    # Pattern to find <answer> tag and capture its content
    answer_pattern = r'^<answer>(.*?)</answer>$'
    match = re.fullmatch(answer_pattern, s, flags=re.DOTALL)

    if not match:
        return 0.0  # Basic tag format is incorrect

    content = match.group(1).strip()
    if not content: # Content inside answer tag is empty
        return 0.0

    try:
        # Attempt to parse the content as a JSON list
        bbox_list = json.loads(content)
        # Check if it's a list and has 4 elements
        if isinstance(bbox_list, list) and len(bbox_list) == 4:
            # Further checks on element types (e.g., all numbers) can be added
            # but the main GIOU calculation part will handle int conversion.
            return 1.0
        else:
            return 0.0  # Parsed, but not a list of 4 elements
    except json.JSONDecodeError:
        return 0.0  # Content is not valid JSON
    except Exception: # Catch any other unexpected error during parsing check
        return 0.0

def check_and_extract(s):
    # Adjusted pattern to extract content from <answer>...</answer> directly
    # This is for the "not_think" scenario.
    pattern = r'^<answer>(.*?)</answer>$'
    match = re.fullmatch(pattern, s, flags=re.DOTALL)
    if match:
        extracted_content = match.group(1).strip()
        # Return 0.0 if extracted content is empty, otherwise the content.
        # This aligns with the expectation that 0.0 from this function means failure.
        return extracted_content if extracted_content else 0.0
    else:
        # Pattern did not match (e.g., s was not just <answer>content</answer>)
        # This case should ideally be caught by check_format if s is predict_str
        return 0.0


def calculate_giou(box1, box2):
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0.0
    
    c_xmin = min(box1[0], box2[0])
    c_ymin = min(box1[1], box2[1])
    c_xmax = max(box1[2], box2[2])
    c_ymax = max(box1[3], box2[3])
    
    c_area = (c_xmax - c_xmin) * (c_ymax - c_ymin)
    
    giou = iou - (c_area - union_area) / c_area
    
    return giou


def track_compute_score_not_think(predict_str: str, ground_truth: str, response_length) -> Dict[str, float]:
    if not predict_str or not ground_truth:
        return {"overall": -1.0, "giou": 0.0, "format_score": 0.0}

    format_score = check_format(predict_str)

    # Extract content within <answer> tag if format is correct
    if format_score == 1.0:
        answer_content = check_and_extract(predict_str)
        if answer_content == 0.0: # check_and_extract returns 0.0 on failure
            return {"overall": -1.0 + (format_score * 0.2), "giou": 0.0, "format_score": format_score} # penalize if extraction fails despite good format
        predict_str_for_giou = answer_content
    else:
        # If format is incorrect, we might still try to parse predict_str directly
        # or penalize heavily. For now, let's assume direct parsing attempt.
        # Or, if strict format adherence is required before GIOU, return low score:
        # return {"overall": format_score * 0.2 - 1.0 * 0.8, "giou": 0.0, "format_score": format_score}
        predict_str_for_giou = predict_str # Fallback or could be an empty string / error marker

    try:
        # Attempt to parse the (potentially extracted) predict_str_for_giou
        pre_bbox = json.loads(predict_str_for_giou) # predict_str_for_giou is like "[10,20,30,40]"
        # ground_truth is like "10,20,30,40"
        gt_coords_str = ground_truth.split(',')
        if len(gt_coords_str) != 4:
            # Handle error if ground_truth is not in "x,y,w,h" format after split
            return {"overall": -1.0 * 0.8 + (format_score * 0.2), "giou": 0.0, "format_score": format_score}
        gt_bbox = [int(c.strip()) for c in gt_coords_str]
        
        if len(pre_bbox) != 4: # Ensure pre_bbox also has 4 coordinates after json.loads
            return {"overall": -1.0 * 0.8 + (format_score * 0.2), "giou": 0.0, "format_score": format_score}

    except (json.JSONDecodeError, ValueError, TypeError): # Catch errors from json.loads, int conversion, or split
        # If parsing fails, GIOU part of score is -1.0
        giou_component = -1.0
        giou_reward_copy = 0.0 # Or -1.0 depending on how you want to report raw GIOU on parse failure
        overall_score = (giou_component * 0.8) + (format_score * 0.2)
        return {"overall": overall_score, "giou": giou_reward_copy, "format_score": format_score}
    
    try:
        giou_reward = calculate_giou(pre_bbox, gt_bbox)
    except:
        giou_reward = -1.0

    giou_reward_copy = giou_reward
    
    # Apply GIOU reward adjustments
    adjusted_giou_reward = giou_reward
    if giou_reward > 0 and giou_reward < 0.4:
        adjusted_giou_reward = 0.0
    elif giou_reward > 0.75 and giou_reward < 0.95:
        adjusted_giou_reward += 0.2
    elif giou_reward > 0.95:
        adjusted_giou_reward += 0.5
    
    # Ensure adjusted_giou_reward does not exceed a reasonable upper bound if necessary, e.g., 1.5 for GIOU + 0.5
    # adjusted_giou_reward = min(adjusted_giou_reward, 1.5) # Example cap

    overall_score = (adjusted_giou_reward * 0.8) + (format_score * 0.2)

    return {
        "overall": overall_score,
        "giou": giou_reward_copy, # Original GIOU before adjustment
        "format_score": format_score,
    }