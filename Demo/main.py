# main.py

import sys
import os
import time
import traceback

# --- 1. 将项目根目录添加到Python路径 ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- 2. 导入各个阶段的主函数 ---
try:
    from scripts.data_preprocessing import main as run_preprocessing
    from scripts.pca_processor import main as run_pca_processing
    from scripts.train_models import main as run_training
    from scripts.analyze_models import main as run_analysis
    from scripts.export_models_to_onnx import main as run_onnx_export
except ImportError as e:
    print(f"错误：无法导入必要的脚本模块。请确保 'scripts' 文件夹及其中的文件都存在。")
    print(f"详细信息: {e}")
    sys.exit(1)


def print_header(title: str):
    """打印一个美观的阶段标题"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"🚀 开始执行阶段: {title}")
    print(f"{line}")


def run_pipeline():
    """
    执行完整的机器学习流水线
    """
    start_total_time = time.time()

    # --- 定义流水线的五个阶段 ---
    pipeline_stages = [
        ("1/5: 数据预处理与拆分", run_preprocessing),
        ("2/5: PCA降维处理", run_pca_processing),
        ("3/5: 模型训练与评估", run_training),
        ("4/5: 模型分析 (SHAP & 预测)", run_analysis),
        ("5/5: ONNX模型导出", run_onnx_export)
    ]

    # --- 按顺序执行每个阶段 ---
    for stage_name, stage_function in pipeline_stages:
        print_header(stage_name)
        start_stage_time = time.time()

        try:
            # 调用当前阶段的主函数
            stage_function()

            end_stage_time = time.time()
            duration = end_stage_time - start_stage_time
            print(f"\n✅ 阶段 '{stage_name}' 成功完成，耗时: {duration:.2f} 秒。")

        except Exception as e:
            # 如果任何一个阶段出错，则立即停止整个流水线
            print(f"\n❌ 在阶段 '{stage_name}' 发生严重错误，流水线终止。")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("\n--- 详细错误追溯 ---")
            traceback.print_exc()
            print("--- 错误追溯结束 ---\n")
            return # 提前退出函数

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print("\n" + "="*80)
    print(" 所有流水线阶段均已成功完成")
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)。")
    print("="*80)


if __name__ == "__main__":
    run_pipeline()
