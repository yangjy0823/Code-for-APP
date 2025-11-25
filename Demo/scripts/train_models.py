# scripts/train_models.py

"""
模型训练、评估与保存脚本 (重构版)
功能:
1.  加载经PCA处理后的训练数据集 (pca_train_data.csv)。
2.  对每个任务(目标变量)、填充方法和模型的组合进行处理。
3.  使用GroupKFold进行交叉验证，评估包括树模型和深度学习模型在内的所有模型。
4.  在完整的训练集上训练最终模型并保存 (sklearn/pytorch)。
5.  生成多Sheet的Excel报告，总结所有组合的评估结果，格式保持不变。
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, precision_score,
                             recall_score, balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef)
import lightgbm as lgb
from tqdm import tqdm
from datetime import datetime

# --- 1. 导入所有必要的模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置、工具函数和模型
from configuration import config
from scripts import utils
from models.deep_learning import LSTMClassifier, GRUClassifier, CNNLSTMClassifier, PyTorchTrainer

# 动态导入torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("警告: PyTorch未安装，将跳过深度学习模型。")

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        """从config模块初始化所有配置"""
        self.config = config
        self.data_config = config.DATA_CONFIG
        self.model_config = config.MODEL_CONFIG
        self.path_config = config.PATH_CONFIG

        self.target_tasks = self.data_config['target_columns']
        self.fill_methods = config.FILLING_STRATEGIES

        tree_models = list(self.model_config['tree_models'].keys())
        deep_models = list(self.model_config['deep_models'].keys()) if DEEP_LEARNING_AVAILABLE else []
        self.models_to_train = [m.replace('_', '-') for m in tree_models + deep_models]

        self.device = self.model_config['pytorch_config']['device'] if DEEP_LEARNING_AVAILABLE else 'cpu'
        print(f"初始化模型训练器，将在 '{self.device}' 设备上运行。")

    def load_feature_data(self):
        """加载PCA处理后的训练数据"""
        print(f"\n加载PCA训练数据: {self.data_config['pca_train_data_path']}")
        try:
            df = pd.read_csv(self.data_config['pca_train_data_path'], encoding='utf-8-sig')
            print(f"数据加载完成，形状: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"错误: 找不到PCA训练数据文件。请先运行 'csv_pca_processor.py'。")
            sys.exit(1)

    def _get_model_instance(self, model_name, **kwargs):
        """根据模型名称和参数获取一个模型实例"""
        if model_name == 'random-forest':
            return RandomForestClassifier(class_weight='balanced', random_state=self.model_config['random_state'], n_jobs=-1)
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier(class_weight='balanced', random_state=self.model_config['random_state'], verbose=-1, n_jobs=-1)
        elif DEEP_LEARNING_AVAILABLE:
            if model_name == 'lstm':
                return LSTMClassifier(**kwargs)
            elif model_name == 'gru':
                return GRUClassifier(**kwargs)
            elif model_name == 'cnn-lstm':
                # CNN-LSTM的参数名可能不同，需要适配
                cnn_kwargs = {
                    'input_size': kwargs['input_size'], 'num_classes': kwargs['num_classes'], 'dropout': kwargs['dropout'],
                    'cnn_filters': 64, 'kernel_size': 3,
                    'lstm_hidden_size': kwargs['hidden_size'], 'lstm_num_layers': kwargs['num_layers']
                }
                return CNNLSTMClassifier(**cnn_kwargs)
        return None

    def evaluate_model_with_cv(self, model_name, X, y, subjects):
        """使用分组交叉验证来评估模型性能 (支持DL模型)"""
        gkf = GroupKFold(n_splits=self.model_config['cv_folds'])
        all_y_true, all_y_pred, all_y_proba = [], [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, subjects)):
            print(f"    - CV Fold {fold+1}/{self.model_config['cv_folds']}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_name in ['random-forest', 'lightgbm']:
                model = self._get_model_instance(model_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

            elif DEEP_LEARNING_AVAILABLE and model_name in ['lstm', 'gru', 'cnn-lstm']:
                num_features = X.shape[1]
                num_classes = len(np.unique(y))
                params = self.model_config['deep_models'][model_name.replace('-', '_')]

                # --- 新增/修改的代码开始 ---
                if model_name == 'cnn-lstm':
                    # 为 CNN-LSTM 构建专属的参数字典
                    model_kwargs = {
                        'input_size': num_features,
                        'hidden_size': params['lstm_hidden_size'][0], # 读取 lstm_hidden_size
                        'num_layers': params['lstm_num_layers'][0],  # 读取 lstm_num_layers
                        'num_classes': num_classes,
                        'dropout': params['dropout'][0],
                        'bidirectional': False
                    }
                else:
                    # 其他模型（lstm, gru）使用通用参数名
                    model_kwargs = {
                        'input_size': num_features, 'hidden_size': params['hidden_size'][0],
                        'num_layers': params['num_layers'][0], 'num_classes': num_classes,
                        'dropout': params['dropout'][0], 'bidirectional': False
                    }
                # --- 新增/修改的代码结束 ---

                model = self._get_model_instance(model_name, **model_kwargs)

                X_train_3d = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val_3d = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])

                train_dataset = TensorDataset(torch.FloatTensor(X_train_3d), torch.LongTensor(y_train.values))
                val_dataset = TensorDataset(torch.FloatTensor(X_val_3d), torch.LongTensor(y_val.values))
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'][0], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'][0], shuffle=False)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'][0])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

                trainer = PyTorchTrainer(model, criterion, optimizer, scheduler, self.device, patience=10)
                trained_model, _ = trainer.fit(train_loader, val_loader, epochs=params['epochs'])

                trained_model.eval()
                with torch.no_grad():
                    outputs = trained_model(torch.FloatTensor(X_val_3d).to(self.device))
                    y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                    y_pred = np.argmax(y_proba, axis=1)
            else:
                continue

            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            all_y_proba.append(y_proba)

        all_y_proba = np.vstack(all_y_proba)
        return self._calculate_all_metrics(all_y_true, all_y_pred, all_y_proba)

    def train_and_save_final_model(self, model_name, X, y, task, fill_method):
        """在所有训练数据上训练最终模型并保存。"""
        print(f"    训练最终模型: {model_name}...")

        if model_name in ['random-forest', 'lightgbm']:
            model = self._get_model_instance(model_name)
            model.fit(X, y)

        elif DEEP_LEARNING_AVAILABLE and model_name in ['lstm', 'gru', 'cnn-lstm']:
            # 对于最终训练，使用早停，需要一个验证集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=self.model_config['random_state'], stratify=y)

            num_features = X.shape[1]
            num_classes = len(np.unique(y))
            params = self.model_config['deep_models'][model_name.replace('-', '_')]

            # --- 新增/修改的代码开始 ---
            if model_name == 'cnn-lstm':
                # 为 CNN-LSTM 构建专属的参数字典
                model_kwargs = {
                    'input_size': num_features,
                    'hidden_size': params['lstm_hidden_size'][0], # 读取 lstm_hidden_size
                    'num_layers': params['lstm_num_layers'][0],  # 读取 lstm_num_layers
                    'num_classes': num_classes,
                    'dropout': params['dropout'][0],
                    'bidirectional': False
                }
            else:
                # 其他模型（lstm, gru）使用通用参数名
                model_kwargs = {
                    'input_size': num_features, 'hidden_size': params['hidden_size'][0],
                    'num_layers': params['num_layers'][0], 'num_classes': num_classes,
                    'dropout': params['dropout'][0], 'bidirectional': False
                }
            # --- 新增/修改的代码结束 --

            model = self._get_model_instance(model_name, **model_kwargs)

            X_train_3d = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_val_3d = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])

            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_3d), torch.LongTensor(y_train.values)), batch_size=params['batch_size'][0], shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_3d), torch.LongTensor(y_val.values)), batch_size=params['batch_size'][0], shuffle=False)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'][0])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            trainer = PyTorchTrainer(model, criterion, optimizer, scheduler, self.device, patience=10)
            model, _ = trainer.fit(train_loader, val_loader, epochs=params['epochs'])
        else:
            model = None

        # 保存模型
        if model:
            ext = ".pth" if DEEP_LEARNING_AVAILABLE and isinstance(model, nn.Module) else ".pkl"
            model_filename = f"{task}_{fill_method}_{model_name}{ext}"
            model_path = os.path.join(self.path_config['models_dir'], model_filename)
            utils.save_model(model, model_path)

    def run_training_pipeline(self):
        """执行完整的训练、评估和保存流程。"""
        df_full_train = self.load_feature_data()
        feature_cols = [col for col in df_full_train.columns if col.startswith('PCA_')]

        all_results_summary = []
        detailed_results = {}
        total_combinations = len(self.target_tasks) * len(self.fill_methods) * len(self.models_to_train)
        pbar = tqdm(total=total_combinations, desc="总进度")

        for task in self.target_tasks:
            if task not in df_full_train.columns or df_full_train[task].isnull().all():
                pbar.update(len(self.fill_methods) * len(self.models_to_train))
                continue

            df_task = df_full_train.dropna(subset=[task])
            df_task[task] = df_task[task].astype(int) # 确保标签是整数

            for fill_method in self.fill_methods:
                task_fill_results = []
                for model_name in self.models_to_train:
                    pbar.set_description(f"处理: {task[:10]}... | {fill_method} | {model_name}")

                    # 准备数据 (此处填充逻辑是可选的，如果PCA后仍有NaN)
                    X = df_task[feature_cols].copy()
                    y = df_task[task]
                    subjects = df_task[self.data_config['subject_id_col']]

                    # 评估
                    metrics = self.evaluate_model_with_cv(model_name, X, y, subjects)

                    # 训练和保存最终模型
                    self.train_and_save_final_model(model_name, X, y, task, fill_method)

                    result_row = {'任务': task, '填充方法': fill_method, '模型': model_name, **metrics}
                    task_fill_results.append(result_row)
                    pbar.update(1)

                sheet_name = f"{task[:15]}_{fill_method}"
                detailed_results[sheet_name] = pd.DataFrame(task_fill_results)
                if task_fill_results:
                    best_result = max(task_fill_results, key=lambda x: x.get('f1_weighted', 0))
                    all_results_summary.append(best_result)
        pbar.close()
        self._generate_excel_report(all_results_summary, detailed_results)

    def _calculate_all_metrics(self, y_true, y_pred, y_proba):
        """计算所有指定的分类模型评估指标。"""
        metrics = {}
        unique_classes = len(np.unique(y_true))

        # --- 核心指标 ---
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Balanced Acc'] = balanced_accuracy_score(y_true, y_pred)

        # --- 基于混淆矩阵的宏平均指标 ---
        metrics['Precision(macro)'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['Recall(macro)'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['F1(macro)'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # --- F1 加权平均 ---
        metrics['F1(weighted)'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # --- Kappa 和 MCC ---
        metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
        try:
            # 当真实标签或预测标签中只有一个类别时，MCC会报错
            metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        except ValueError:
            metrics['MCC'] = 0.0

        # --- ROC AUC (处理二分类和多分类) ---
        try:
            if y_proba is not None and unique_classes > 1:
                if unique_classes > 2:  # 多分类
                    metrics['ROC_AUC(macro_OvR)'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                else:  # 二分类
                    metrics['ROC_AUC(macro_OvR)'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['ROC_AUC(macro_OvR)'] = 0.0
        except ValueError:
            metrics['ROC_AUC(macro_OvR)'] = 0.0

        return metrics

    def _generate_excel_report(self, summary_data, detailed_data):
        # (此辅助函数与原版一致，此处省略以保持简洁，实际使用时请保留)
        if not summary_data:
            print("没有可生成的报告数据。")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"model_evaluation_report_{timestamp}.xlsx"
        report_path = os.path.join(self.path_config['evaluation_reports_dir'], report_filename)

        print(f"\n正在生成Excel报告: {report_path}")
        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='总体最佳模型对比', index=False)

                for sheet_name, data_df in detailed_data.items():
                    safe_sheet_name = sheet_name.replace(':', '').replace('\\', '').replace('/', '').replace('?', '').replace('*', '').replace('[', '').replace(']', '')[:31]
                    data_df.sort_values(by='F1(weighted)', ascending=False, inplace=True)
                    data_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

            print("Excel报告生成成功!")
        except Exception as e:
            print(f"错误: Excel报告生成失败 - {e}")

def main():
    # 确保所有输出目录都存在
    for dir_path in [config.PATH_CONFIG['models_dir'], config.PATH_CONFIG['evaluation_reports_dir'], config.PATH_CONFIG['checkpoints_dir']]:
        os.makedirs(dir_path, exist_ok=True)

    try:
        trainer = ModelTrainer()
        trainer.run_training_pipeline()
        print("\n所有训练任务完成。")
    except Exception as e:
        import traceback
        print(f"\n脚本执行过程中发生严重错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
