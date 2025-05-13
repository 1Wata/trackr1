import subprocess
import os
import sys

# --- 配置 ---
# 假设此脚本位于 /data1/lihaobo/track_r1/
# 并且 rft_tracking_inference.py 位于 ./dataset_interface/
# 并且 analysis_results.py 位于 ./ (与此脚本相同的目录)

RFT_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "dataset_interface", "rft_tracking_inference.py")
# 假设 analysis_results.py 与此脚本在同一目录中
# 如果 analysis_results.py 在其他位置，请相应地更新路径
ANALYSIS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "analysis_results.py") 

MODEL_PATH = "/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-3"

BASE_PIPELINE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tracking_pipeline_results") 
SAVE_VIS = False # 是否在推理过程中保存可视化结果
MAX_NEW_TOKENS = 2048


DATASETS = ["lasot", "OTB_lang", "TNL2k"]
# 用于分析脚本中的跟踪器名称前缀
TRACKER_NAME_PREFIX = "RFT_Qwen2.5_VL_3B_Instruct" 

# --- 主要逻辑 ---
def run_command(command_list):
    """运行命令并打印其输出。"""
    print(f"正在执行: {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, capture_output=True, text=True, check=False)
        if process.stdout:
            print("STDOUT:\n", process.stdout)
        if process.stderr:
            print("STDERR:\n", process.stderr)
        if process.returncode != 0:
            print(f"命令执行失败，返回码 {process.returncode}")
            return False
        print("命令执行成功。")
        return True
    except FileNotFoundError:
        print(f"错误: 找不到脚本。请检查路径: {' '.join(command_list)}")
        return False
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
        return False

def main():
    os.makedirs(BASE_PIPELINE_OUTPUT_DIR, exist_ok=True)
    rft_base_output_dir = os.path.join(BASE_PIPELINE_OUTPUT_DIR, "rft_inference_outputs")
    os.makedirs(rft_base_output_dir, exist_ok=True)

    for dataset in DATASETS:
        print(f"\n--- 正在处理数据集: {dataset} ---")

        # 1. 运行 RFT 跟踪推理
        print(f"\n正在为 {dataset} 运行 RFT 跟踪推理...")
        
        cmd_rft = [
            sys.executable, RFT_SCRIPT_PATH,
            "--model_path", MODEL_PATH,
            "--dataset_name", dataset,
            "--output_dir", rft_base_output_dir, # rft_script 会在此目录下创建 <dataset_name> 子目录
            "--save_vis", str(SAVE_VIS).lower(),
            "--max_new_tokens", str(MAX_NEW_TOKENS)
            # 如果需要，可以在此处添加 --gap_list，例如: "--gap_list", "1", "10"
            # 如果需要，可以添加 --single_process
        ]
        
        if not run_command(cmd_rft):
            print(f"数据集 {dataset} 的 RFT 推理失败。跳过分析。")
            continue

        
        tracker_results_path = os.path.join(rft_base_output_dir, dataset)
        # 为分析脚本创建一个唯一的跟踪器名称
        tracker_name_for_analysis = f"{TRACKER_NAME_PREFIX}_{dataset}" 

        print(f"\n正在为 {dataset} 运行分析...")
        # 假设 analysis_results.py 接受 --dataset, --tracker_name, --tracker_path 参数
        # 请根据您的 analysis_results.py 脚本调整这些参数
        cmd_analysis = [
            sys.executable, ANALYSIS_SCRIPT_PATH,
            "--dataset", dataset, # 或者 --dataset_name，具体取决于您的脚本
            "--tracker_name", tracker_name_for_analysis,
            "--tracker_path", tracker_results_path
            # 为 analysis_results.py 添加其他必要的参数
        ]

        if not os.path.exists(ANALYSIS_SCRIPT_PATH):
            print(f"错误: 分析脚本未找到于: {ANALYSIS_SCRIPT_PATH}")
            print("请确保 analysis_results.py 存在于预期的位置，或者更新脚本中的 ANALYSIS_SCRIPT_PATH。")
            continue

        if not run_command(cmd_analysis):
            print(f"数据集 {dataset} 的分析失败。")
            continue
        
        print(f"--- 数据集 {dataset} 已成功处理和分析 ---")

    print("\n所有数据集处理完毕。")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()