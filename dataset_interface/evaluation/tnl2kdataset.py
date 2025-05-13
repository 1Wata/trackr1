import os

import numpy as np
from evaluation.data import Sequence, BaseDataset, SequenceList
from evaluation.utils.load_text import load_text, load_str
from utils.utils import clean_string

############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        # self.base_path = os.path.join(self.env_settings.tnl2k_path, 'test')
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_name = sequence_name.split('/')[-1]
        # class_name = seq_name
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        if self.dir_type == 'one-level':
            return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))
        elif self.dir_type == 'two-level':
            return Sequence(seq_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        subset_list = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
        
        # one-level directory
        if len(subset_list) > 9:
            self.dir_type = 'one-level'
            return sorted(subset_list) 
        
        # two-level or deeper directory
        self.dir_type = 'two-level'
        for category_name in subset_list: # e.g., category_name is a first-level directory like 'Animal'
            current_category_path = os.path.join(self.base_path, category_name)
            
            for item_in_category in os.listdir(current_category_path): # e.g., item_in_category is 'dog_run_01' or 'Outdoor_scenes'
                path_level2_abs = os.path.join(current_category_path, item_in_category) # Absolute path to the item
                
                if not os.path.isdir(path_level2_abs):
                    continue

                # Check for 'imgs' folder at the current level (e.g., base_path/category_name/item_in_category/imgs)
                imgs_path_at_level2 = os.path.join(path_level2_abs, 'imgs')
                
                if os.path.isdir(imgs_path_at_level2):
                    # This is a sequence directory. Add its relative path.
                    # e.g., 'Animal/dog_run_01'
                    sequence_list.append(os.path.join(category_name, item_in_category))
                else:
                    # If 'imgs' not found, item_in_category might be a sub-category.
                    # Look one level deeper.
                    # path_level2_abs is now treated as a sub-category path.
                    # e.g., base_path/category_name/Outdoor_scenes/
                    sub_category_path_abs = path_level2_abs
                    for item_in_sub_category in os.listdir(sub_category_path_abs): # e.g., item_in_sub_category is 'forest_hike_03'
                        path_level3_abs = os.path.join(sub_category_path_abs, item_in_sub_category) # Absolute path to the potential sequence

                        if not os.path.isdir(path_level3_abs):
                            continue
                        
                        # Check for 'imgs' folder at the deeper level
                        # e.g., base_path/category_name/item_in_category/item_in_sub_category/imgs
                        imgs_path_at_level3 = os.path.join(path_level3_abs, 'imgs')
                        if os.path.isdir(imgs_path_at_level3):
                            # This is a sequence directory at a deeper level. Add its relative path.
                            # e.g., 'Animal/Outdoor_scenes/forest_hike_03'
                            sequence_list.append(os.path.join(category_name, item_in_category, item_in_sub_category))
        
        sequence_list = sorted(sequence_list)

        return sequence_list


class TNL2k_LangDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.tnl2k_path, 'test')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_name = sequence_name.split('/')[-1]
        class_name = seq_name
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)
        text_dsp = clean_string(text_dsp)
        
        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        target_class = class_name
        if self.dir_type == 'one-level':
            return Sequence(sequence_name, frames_list, 'tnl2k_lang', ground_truth_rect.reshape(-1, 4),
                            text_description=text_dsp, object_class=target_class)
        elif self.dir_type == 'two-level':
            return Sequence(seq_name, frames_list, 'tnl2k_lang', ground_truth_rect.reshape(-1, 4),
                        text_description=text_dsp, object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        subset_list = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
        
        # one-level directory
        if len(subset_list) > 9:
            self.dir_type = 'one-level'
            return sorted(subset_list) 
        
        # two-level or deeper directory
        self.dir_type = 'two-level'
        for category_name in subset_list: # e.g., category_name is a first-level directory like 'Animal'
            current_category_path = os.path.join(self.base_path, category_name)
            
            for item_in_category in os.listdir(current_category_path): # e.g., item_in_category is 'dog_run_01' or 'Outdoor_scenes'
                path_level2_abs = os.path.join(current_category_path, item_in_category) # Absolute path to the item
                
                if not os.path.isdir(path_level2_abs):
                    continue

                # Check for 'imgs' folder at the current level (e.g., base_path/category_name/item_in_category/imgs)
                imgs_path_at_level2 = os.path.join(path_level2_abs, 'imgs')
                
                if os.path.isdir(imgs_path_at_level2):
                    # This is a sequence directory. Add its relative path.
                    # e.g., 'Animal/dog_run_01'
                    sequence_list.append(os.path.join(category_name, item_in_category))
                else:
                    # If 'imgs' not found, item_in_category might be a sub-category.
                    # Look one level deeper.
                    # path_level2_abs is now treated as a sub-category path.
                    # e.g., base_path/category_name/Outdoor_scenes/
                    sub_category_path_abs = path_level2_abs
                    for item_in_sub_category in os.listdir(sub_category_path_abs): # e.g., item_in_sub_category is 'forest_hike_03'
                        path_level3_abs = os.path.join(sub_category_path_abs, item_in_sub_category) # Absolute path to the potential sequence

                        if not os.path.isdir(path_level3_abs):
                            continue
                        
                        # Check for 'imgs' folder at the deeper level
                        # e.g., base_path/category_name/item_in_category/item_in_sub_category/imgs
                        imgs_path_at_level3 = os.path.join(path_level3_abs, 'imgs')
                        if os.path.isdir(imgs_path_at_level3):
                            # This is a sequence directory at a deeper level. Add its relative path.
                            # e.g., 'Animal/Outdoor_scenes/forest_hike_03'
                            sequence_list.append(os.path.join(category_name, item_in_category, item_in_sub_category))
        
        sequence_list = sorted(sequence_list)

        return sequence_list
