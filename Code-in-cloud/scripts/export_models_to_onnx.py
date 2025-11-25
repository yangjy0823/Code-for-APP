# scripts/export_models_to_onnx.py

"""
ONNX模型导出器 (重构版)
从我们流水线已保存的模型文件中，导出所有最佳模型到ONNX格式。
...
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm  # --- 核心修正：新增导入tqdm ---

# --- 1. 导入所有必要的模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration import config
from scripts import utils
from models.deep_learning import LSTMClassifier, GRUClassifier, CNNLSTMClassifier

# 深度学习相关
try:
    import torch
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("警告: PyTorch未安装，深度学习模型将跳过")

# ONNX导出相关
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxmltools
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: onnx, onnxruntime, skl2onnx或onnxmltools未安装，无法执行导出。")

warnings.filterwarnings('ignore')


class ONNXModelExporter:
    def __init__(self):
        self.config = config
        self.path_config = config.PATH_CONFIG
        self.model_config = config.MODEL_CONFIG
        self.output_dir = os.path.join(self.path_config['models_dir'], 'onnx')
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = self.model_config['pytorch_config']['device'] if DEEP_LEARNING_AVAILABLE else 'cpu'
        print(f"ONNX导出器初始化完成，输出目录: {self.output_dir}")

    def _get_export_info(self):
        """从评估报告和PCA模型中获取所有必要信息"""
        # 1. 查找最新的评估报告
        latest_report = utils.find_latest_file(self.path_config['evaluation_reports_dir'], ".xlsx")
        if not latest_report:
            raise FileNotFoundError(f"在 {self.path_config['evaluation_reports_dir']} 中找不到评估报告。")
        print(f"使用评估报告: {os.path.basename(latest_report)}")

        # 2. 从报告中提取最佳模型组合
        df = pd.read_excel(latest_report, sheet_name='总体最佳模型对比')
        best_models = []
        for _, row in df.iterrows():
            best_models.append({'任务': row['任务'], '填充方法': row['填充方法'], '模型': row['模型'].replace('_', '-')})

        # 3. 加载PCA对象以获取特征数量
        with open(self.path_config['pca_model_path'], 'rb') as f:
            preprocessors = pickle.load(f)
        num_features = preprocessors['pca'].n_components_
        print(f"从PCA模型中获取到输入特征维度: {num_features}")

        return best_models, num_features

    def _load_best_model(self, combo: dict, num_features: int, num_classes: int):
        """加载单个最佳模型（与analyze_models.py中的逻辑相同）"""
        task, fill_method, model_name = combo['任务'], combo['填充方法'], combo['模型']
        ext = ".pth" if model_name in ['lstm', 'gru', 'cnn-lstm'] else ".pkl"
        model_filename = f"{task}_{fill_method}_{model_name}{ext}"
        model_path = os.path.join(self.path_config['models_dir'], model_filename)

        if not os.path.exists(model_path):
             print(f"    [警告] 找不到模型文件: {model_filename}，跳过。")
             return None

        print(f"    正在加载模型: {model_filename}")

        if ext == ".pth":
            params = self.model_config['deep_models'][model_name.replace('-', '_')]
            model_kwargs = {
                'input_size': num_features, 'num_classes': num_classes, 'hidden_size': params['hidden_size'][0],
                'num_layers': params['num_layers'][0], 'dropout': params['dropout'][0], 'bidirectional': False
            }
            if model_name == 'lstm': model_instance = LSTMClassifier(**model_kwargs)
            elif model_name == 'gru': model_instance = GRUClassifier(**model_kwargs)
            else:
                cnn_kwargs = {
                    'input_size': num_features, 'num_classes': num_classes, 'dropout': params['dropout'][0],
                    'cnn_filters': params['cnn_filters'][0], 'kernel_size': params['kernel_sizes'][0],
                    'lstm_hidden_size': params['lstm_hidden_size'][0], 'lstm_num_layers': params['lstm_num_layers'][0]
                }
                model_instance = CNNLSTMClassifier(**cnn_kwargs)
            return utils.load_model(model_path, model_instance=model_instance)
        else:
            return utils.load_model(model_path)

    def _export_to_onnx(self, model, model_type, num_features, onnx_path):
        """将加载好的模型对象导出到ONNX"""
        model_type_lower = model_type.lower().replace('-', '_')

        if 'random_forest' in model_type_lower:
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

        elif 'lightgbm' in model_type_lower:
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=12)
            onnxmltools.utils.save_model(onnx_model, onnx_path)

        elif DEEP_LEARNING_AVAILABLE and isinstance(model, torch.nn.Module):
            model.to(self.device).eval()
            dummy_input = torch.randn(1, 1, num_features, device=self.device)
            torch.onnx.export(
                model, dummy_input, onnx_path, export_params=True, opset_version=12,
                do_constant_folding=True, input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        else:
            print(f"    不支持的模型类型 '{model_type}' 或相关库未安装。")
            return False

        return True

    def run_export(self):
        """主执行函数"""
        if not ONNX_AVAILABLE:
            print("错误: 缺少ONNX相关库，无法执行导出。请运行: pip install onnx onnxruntime skl2onnx onnxmltools")
            return

        try:
            best_models, num_features = self._get_export_info()
        except Exception as e:
            print(f"错误: 准备导出信息失败 - {e}")
            return

        print("\n" + "="*60)
        print(f"开始导出 {len(best_models)} 个最佳模型到ONNX...")
        print("="*60)

        success_count = 0
        for combo in tqdm(best_models, desc="导出ONNX模型"):
            task, model_type = combo['任务'], combo['模型']
            num_classes = 2

            try:
                model = self._load_best_model(combo, num_features, num_classes)
                if model is None:
                    continue

                onnx_filename = f"{task}_{combo['填充方法']}_{model_type}.onnx"
                onnx_path = os.path.join(self.output_dir, onnx_filename)

                if self._export_to_onnx(model, model_type, num_features, onnx_path):
                    print(f"    ✅ 导出成功: {onnx_filename}")
                    success_count += 1
                else:
                    print(f"    ❌ 导出失败: {onnx_filename}")

            except Exception as e:
                import traceback
                print(f"    ❌ 处理组合 {combo} 时发生意外错误: {e}")
                traceback.print_exc()

        print("\n" + "="*60)
        print("ONNX导出流程完成")
        print(f"成功导出: {success_count}/{len(best_models)}")
        print("="*60)

        if success_count > 0:
            self._validate_onnx_models(num_features)

    def _validate_onnx_models(self, num_features):
        """验证导出的ONNX模型是否可以加载和运行"""
        print("\n开始验证导出的ONNX模型...")
        onnx_files = [f for f in os.listdir(self.output_dir) if f.endswith('.onnx')]

        for onnx_file in onnx_files:
            try:
                onnx_path = os.path.join(self.output_dir, onnx_file)
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape

                if len(input_shape) == 3: # PyTorch model
                    test_input = np.random.randn(1, 1, num_features).astype(np.float32)
                else: # Sklearn model
                    test_input = np.random.randn(1, num_features).astype(np.float32)

                session.run(None, {input_name: test_input})
                print(f"    ✅ {onnx_file} - 验证通过")
            except Exception as e:
                print(f"    ❌ {onnx_file} - 验证失败: {e}")

def main():
    exporter = ONNXModelExporter()
    exporter.run_export()

if __name__ == "__main__":
    main()
