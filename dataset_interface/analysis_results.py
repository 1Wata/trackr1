import os
import sys
import torch
import numpy as np
from collections import namedtuple


env_path = os.path.dirname(os.path.abspath(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

from evaluation import get_dataset
from analysis.extract_results import extract_results
from analysis.plot_results import get_auc_curve, get_prec_curve
from evaluation.environment import env_settings
import argparse

SimpleTracker = namedtuple('SimpleTracker', ['name', 'parameter_name', 'run_id', 'results_dir', 'display_name'])

def evaluate_direct(results_path, dataset_name):
    """直接评估预测结果
    
    Args:
        results_path: 预测结果txt文件所在的目录 (e.g., output_dir/lasot)
        dataset_name: 数据集名称 (e.g., lasot)
    """

    dataset = get_dataset(dataset_name)
    # print(f"成功加载数据集: {dataset_name}, 共有 {len(dataset)} 个序列")
    print(f"sucessfully load dataset: {dataset_name}, total {len(dataset)} sequences")

    
    # parameter_name_for_report 用于报告中的显示，可以保持为数据集名称
    parameter_name_for_report = os.path.basename(results_path)
    # tracker_base_results_dir 是包含各数据集结果文件夹的父目录 (e.g., output_dir)
    tracker_base_results_dir = os.path.dirname(results_path)
    
    tracker = SimpleTracker(
        name='custom_tracker',
        parameter_name="",  # 设置为空字符串，以避免在路径中创建额外的层级
        run_id=None,
        results_dir=tracker_base_results_dir,  # 指向包含数据集文件夹的目录
        display_name=f'CustomTracker_{parameter_name_for_report}' # 或者 'CustomTracker'
    )
    
    report_name = f"{dataset_name}_eval" # 或者 f"{parameter_name_for_report}_eval"
    eval_data = extract_results(
        trackers=[tracker], 
        dataset=dataset, 
        report_name=report_name,
        skip_missing_seq=True,  # 跳过缺失的序列
        plot_bin_gap=0.05,
        exclude_invalid_frames=False
    )
    
    print("\n evaluate done!")
    

    if eval_data:
        valid_sequence = torch.tensor(eval_data['valid_sequence'])
        if not valid_sequence.any():
            print("warning: no valid sequences found.")
            return
        
        # AUC (Success) 计算
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        
        # Precision 计算
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        
        # Normalized Precision 计算
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        
        print(f"dataset name: {dataset_name}")
        print(f"evaled sequence num: {valid_sequence.sum().item()} / {len(dataset)}")
        print(f"Success (AUC): {auc.item():.4f}")
        print(f"Precision: {prec_score.item():.4f}")
        print(f"Normalized Precision: {norm_prec_score.item():.4f}")
        
        
        output_summary_dir = os.path.join(os.path.dirname(results_path), report_name)
        os.makedirs(output_summary_dir, exist_ok=True) #确保目录存在
        summary_txt_file_path = os.path.join(output_summary_dir, "evaluation_summary.txt")

        with open(summary_txt_file_path, "w") as f:
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write(f"Tracker Parameter Name: {parameter_name_for_report}\n") # 使用 for_report 版本
            f.write(f"Evaluated Sequences: {valid_sequence.sum().item()} / {len(dataset)}\n")
            f.write(f"Success (AUC): {auc.item():.4f}\n")
            f.write(f"Precision: {prec_score.item():.4f}\n")
            f.write(f"Normalized Precision: {norm_prec_score.item():.4f}\n")
        
        print(f"\nEvaluation summary saved to: {summary_txt_file_path}")



def fix_dir(dir_path):
    for sub_dir_name in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        if os.path.isdir(sub_dir_path):
            predictions_file_path = os.path.join(sub_dir_path, "predictions.txt")
            if os.path.exists(predictions_file_path):
                new_file_name = f"{sub_dir_name}.txt"
                new_file_path = os.path.join(dir_path, new_file_name)

                os.rename(predictions_file_path, new_file_path)
                print(f"Renamed: {predictions_file_path} to {new_file_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate tracking results directly from prediction files.')
    parser.add_argument('results_path', type=str, help='Path to the directory containing prediction txt files.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to evaluate on.')
    
    args = parser.parse_args()
    fix_dir(args.results_path)  # 修复目录结构
    my_results_path = args.results_path
    my_dataset_name = args.dataset_name
    
    evaluate_direct(my_results_path, my_dataset_name)

