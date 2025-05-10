# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import re # Added for bbox scaling in prompt

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
            len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]

        # --- Start of BBox Scaling Logic ---
        original_image_size = None
        resized_image_size = None
        scale_x, scale_y = 1.0, 1.0

        if self.image_key in example and example[self.image_key]:
            # Access image data to determine scaling factors
            # We assume example[self.image_key] is a list of image data (e.g., dicts with "bytes" or raw bytes)
            raw_image_data_list = example[self.image_key] # Do not pop yet

            if isinstance(raw_image_data_list, list) and raw_image_data_list:

                first_raw_item = raw_image_data_list[0]
                temp_pil_img_orig = None
                if isinstance(first_raw_item, dict) and "bytes" in first_raw_item:
                    temp_pil_img_orig = Image.open(BytesIO(first_raw_item["bytes"]))
                elif isinstance(first_raw_item, bytes): # Handle if raw data is just bytes
                    temp_pil_img_orig = Image.open(BytesIO(first_raw_item))
                # Add other potential raw formats if necessary

                if temp_pil_img_orig:
                    original_image_size = (temp_pil_img_orig.width, temp_pil_img_orig.height)
                    # Process a copy of the first image to get resized dimensions
                    # self.process_image is defined in ImageProcessMixin
                    first_processed_pil_img = self.process_image(temp_pil_img_orig.copy())
                    resized_image_size = (first_processed_pil_img.width, first_processed_pil_img.height)

                    if original_image_size[0] > 0 and original_image_size[1] > 0:
                        scale_x = resized_image_size[0] / original_image_size[0]
                        scale_y = resized_image_size[1] / original_image_size[1]
                    else:
                        print(f"Warning (idx {index}): Invalid original image dimensions {original_image_size}. Bbox scaling factors set to 1.")
                        scale_x, scale_y = 1.0, 1.0
                    
                    # Store sizes in example if needed elsewhere, though not strictly required by prompt
                    example["original_image_size"] = original_image_size
                    example["resized_image_size"] = resized_image_size
                else:
                    print(f"Warning (idx {index}): Could not open first image to determine dimensions. Bbox scaling skipped.")
                    scale_x, scale_y = 1.0, 1.0


            else: # No images in the list
                scale_x, scale_y = 1.0, 1.0


            # Apply scaling if factors are not 1.0 (i.e., resize happened and was measurable)
            if scale_x != 1.0 or scale_y != 1.0:
                # 1. Scale Ground Truth Bbox (in example[self.answer_key])
                # Assumes GT bbox is "x1,y1,x2,y2" string
                if self.answer_key in example and example[self.answer_key] and isinstance(example[self.answer_key], str):
                    gt_bbox_str = example[self.answer_key]

                    coords_str_list = gt_bbox_str.split(',')
                    if len(coords_str_list) == 4:
                        coords_orig = [int(c.strip()) for c in coords_str_list]
                        x1_orig, y1_orig, x2_orig, y2_orig = coords_orig

                        x1_s = round(x1_orig * scale_x)
                        y1_s = round(y1_orig * scale_y)
                        x2_s = round(x2_orig * scale_x)
                        y2_s = round(y2_orig * scale_y)

                        if resized_image_size and resized_image_size[0] > 0 and resized_image_size[1] > 0:
                            x1_s = max(0, min(x1_s, resized_image_size[0] - 1))
                            y1_s = max(0, min(y1_s, resized_image_size[1] - 1))
                            x2_s = max(0, min(x2_s, resized_image_size[0] - 1))
                            y2_s = max(0, min(y2_s, resized_image_size[1] - 1))
                        
                        if x1_s > x2_s: x1_s, x2_s = x2_s, x1_s # Ensure x1 <= x2
                        if y1_s > y2_s: y1_s, y2_s = y2_s, y1_s # Ensure y1 <= y2
                        
                        example[self.answer_key] = f"{x1_s},{y1_s},{x2_s},{y2_s}"
                        # else: Malformed GT bbox string, leave as is or log warning

                # 2. Scale Bboxes within the prompt string (example[self.prompt_key])
                # Assumes bboxes in prompt are like "[x1,y1,x2,y2]"
                if self.prompt_key in example and example[self.prompt_key] and isinstance(example[self.prompt_key], str):
                    current_prompt_str = example[self.prompt_key]

                    def scale_prompt_bbox_callback(match_obj):

                        # match_obj.groups() will be (num_str1, num_str2, num_str3, num_str4)
                        coords_orig_str = match_obj.groups()
                        coords_orig = [int(c.strip()) for c in coords_orig_str]
                        
                        x1_orig, y1_orig, x2_orig, y2_orig = coords_orig

                        x1_s = round(x1_orig * scale_x)
                        y1_s = round(y1_orig * scale_y)
                        x2_s = round(x2_orig * scale_x)
                        y2_s = round(y2_orig * scale_y)

                        if resized_image_size and resized_image_size[0] > 0 and resized_image_size[1] > 0:
                            x1_s = max(0, min(x1_s, resized_image_size[0] - 1))
                            y1_s = max(0, min(y1_s, resized_image_size[1] - 1))
                            x2_s = max(0, min(x2_s, resized_image_size[0] - 1))
                            y2_s = max(0, min(y2_s, resized_image_size[1] - 1))

                        if x1_s > x2_s: x1_s, x2_s = x2_s, x1_s # Ensure x1 <= x2
                        if y1_s > y2_s: y1_s, y2_s = y2_s, y1_s # Ensure y1 <= y2
                        
                        return f"[{x1_s},{y1_s},{x2_s},{y2_s}]"

                    # Regex to find "[num, num, num, num]" with optional spaces
                    scaled_prompt_str = re.sub(
                        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
                        scale_prompt_bbox_callback,
                        current_prompt_str
                    )
                    example[self.prompt_key] = scaled_prompt_str
        # --- End of BBox Scaling Logic ---

        messages = self._build_messages(example) # Uses example with potentially scaled prompt_key

        # Original logic for multimodal data
        if self.image_key in example: # Check again as example[self.image_key] might be empty
            # The prompt variable below will be generated from 'messages' which are based on the
            # potentially scaled example[self.prompt_key]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # Process all images. example[self.image_key] is popped here.
            # The first image was already processed (as a copy) to get resized_image_size.
            # self.process_image will handle each item.
            images = [self.process_image(img_data) for img_data in example.pop(self.image_key)]
            
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images} # 'images' are processed PIL ImageObjects
            example["multi_modal_inputs"] = dict(model_inputs)
        else: # Original logic for text-only data
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # The rest of the method remains unchanged
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"), # Uses model_inputs from above
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        # 'prompt' here is the one generated after messages (which are based on scaled prompt_key)
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                # Ensure index is part of the error message for better debugging
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} for example index {index} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        # example[self.answer_key] was potentially scaled, so ground_truth will use the scaled version.
        example["ground_truth"] = example.pop(self.answer_key) 
        return example
