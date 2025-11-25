# scripts/analyze_models.py

"""
模型分析与预测脚本 (集成增强版SHAP可视化 - 终极修复版)

功能:
1.  读取最新的模型评估报告，找出每个任务的最佳模型组合。
2.  加载在PCA流程中保存的 preprocessor (scaler & pca 对象)。
3.  加载独立的、经PCA处理的测试数据集 (pca_test_data.csv)。
4.  智能加载预训练好的最佳模型 (支持Sklearn和PyTorch)。
5.  对每个最佳模型进行全面的SHAP可解释性分析 (支持TreeExplainer和DeepExplainer)，
    并保存三种可视化图片：Summary Plot, Dependence Plot, 和 Force Plot。
6.  使用模型对测试集进行预测，并生成与原始格式一致的多Sheet Excel预测报告。
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
from tqdm import tqdm
import pickle
import gc

# --- 1. 导入所有必要的模块 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configuration import config
from scripts import utils
from models.deep_learning import LSTMClassifier, GRUClassifier, CNNLSTMClassifier

# 动态导入torch并设置matplotlib中文字体
try:
    import torch
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("警告: PyTorch未安装，将无法加载和分析深度学习模型。")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# --- SHAP配置 ---
SHAP_CONFIG = {
    'max_samples_for_shap': 200,
    'top_features_dependence': 5,
    'n_force_plot_samples': 3
}

OUTPUT_CONFIG = {
    'image_dpi': 300
}


class ModelAnalyzer:
    def __init__(self):
        print("初始化模型分析器...")
        self.config = config
        self.path_config = config.PATH_CONFIG
        self.model_config = config.MODEL_CONFIG

        os.makedirs(self.path_config['shap_analysis_dir'], exist_ok=True)
        os.makedirs(self.path_config['predictions_dir'], exist_ok=True)

        try:
            latest_report = utils.find_latest_file(self.path_config['evaluation_reports_dir'], ".xlsx")
            if latest_report is None:
                raise FileNotFoundError(f"在 {self.path_config['evaluation_reports_dir']} 中找不到评估报告。请先运行 train_models.py。")

            print(f"使用评估报告: {os.path.basename(latest_report)}")
            self.best_combinations = self._extract_best_models(latest_report)
            print(f"已提取 {len(self.best_combinations)} 个最佳模型组合进行分析。")
        except Exception as e:
            print(f"错误：无法初始化分析器 - {e}")
            self.best_combinations = []

    def _extract_best_models(self, report_path: str) -> list:
        df = pd.read_excel(report_path, sheet_name='总体最佳模型对比')
        required_cols = ['任务', '填充方法', '模型']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("报告缺少必要的列（任务, 填充方法, 模型）。")

        combinations = df[required_cols].to_dict('records')
        for combo in combinations:
            combo['模型'] = combo['模型'].replace('_', '-')
        return combinations

    def _load_data_and_preprocessors(self):
        print("\n加载测试数据和预处理器...")
        test_df = pd.read_csv(self.config.DATA_CONFIG['pca_test_data_path'])
        print(f"测试数据加载完成，形状: {test_df.shape}")

        with open(self.path_config['pca_model_path'], 'rb') as f:
            preprocessors = pickle.load(f)
        print(f"预处理器(Scaler, PCA)加载完成。")

        return test_df, preprocessors

    def _load_best_model(self, combo: dict, num_features: int, num_classes: int):
        task, fill_method, model_name = combo['任务'], combo['填充方法'], combo['模型']

        ext = ".pth" if model_name in ['lstm', 'gru', 'cnn-lstm'] else ".pkl"
        model_filename = f"{task}_{fill_method}_{model_name}{ext}"
        model_path = os.path.join(self.path_config['models_dir'], model_filename)
        print(f"      正在加载模型: {model_filename}")

        if not os.path.exists(model_path):
            print(f"      [警告] 模型文件不存在: {model_path}，跳过此组合。")
            return None

        if not DEEP_LEARNING_AVAILABLE and ext == ".pth":
            print("      警告: PyTorch未安装，无法加载深度学习模型，跳过。")
            return None

        model_instance = None
        if ext == ".pth":
            params = self.model_config['deep_models'][model_name.replace('-', '_')]
            if model_name == 'lstm' or model_name == 'gru':
                model_kwargs = {
                    'input_size': num_features, 'num_classes': num_classes,
                    'hidden_size': params['hidden_size'][0], 'num_layers': params['num_layers'][0],
                    'dropout': params['dropout'][0], 'bidirectional': False
                }
                model_instance = LSTMClassifier(**model_kwargs) if model_name == 'lstm' else GRUClassifier(**model_kwargs)
            else: # cnn-lstm
                cnn_kwargs = {
                    'input_size': num_features, 'num_classes': num_classes, 'dropout': params['dropout'][0],
                    'cnn_filters': params['cnn_filters'][0], 'kernel_size': params['kernel_sizes'][0],
                    'lstm_hidden_size': params['lstm_hidden_size'][0], 'lstm_num_layers': params['lstm_num_layers'][0]
                }
                model_instance = CNNLSTMClassifier(**cnn_kwargs)

        model = utils.load_model(model_path, model_instance=model_instance)
        return model

    def _generate_shap_visualizations(self, model, model_name: str, X_test_df: pd.DataFrame, combo: dict):
        print("      生成全面的SHAP可视化...")
        task, fill_method = combo['任务'], combo['填充方法']
        output_dir = os.path.join(self.path_config['shap_analysis_dir'], task, fill_method, model_name)
        os.makedirs(output_dir, exist_ok=True)

        max_samples = SHAP_CONFIG['max_samples_for_shap']
        X_sample_df = shap.sample(X_test_df, max_samples) if len(X_test_df) > max_samples else X_test_df

        # --- 终极修复 1: 数据净化 - 将所有输入数据转换为最纯粹的 NumPy 格式 ---
        X_sample_np = X_sample_df.astype(np.float64).values
        feature_names = X_sample_df.columns.tolist()

        try:
            explainer = None
            shap_values = None

            if DEEP_LEARNING_AVAILABLE and isinstance(model, torch.nn.Module):
                print("         使用 DeepExplainer for PyTorch...")
                model.eval()
                explainer = shap.DeepExplainer(model, torch.from_numpy(X_sample_np).float())
                shap_values = explainer.shap_values(torch.from_numpy(X_sample_np).float())
            else:
                print("         使用 TreeExplainer or KernelExplainer...")
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample_df) # TreeExplainer通常对DataFrame处理得更好
                except Exception:
                    print("         TreeExplainer失败，回退到KernelExplainer...")
                    background_summary = shap.kmeans(X_sample_np, 10)
                    explainer = shap.KernelExplainer(model.predict_proba, background_summary)
                    shap_values = explainer.shap_values(X_sample_np)

            # --- 终极修复: 统一的维度处理（参考工作版本comprehensive_shap_main_best_models.py）---
            # 确保维度匹配
            if len(shap_values.shape) > 2:
                # 多分类情况，取第一个类别的SHAP值
                shap_values_2d = shap_values[:, :, 0] if shap_values.shape[2] > 0 else shap_values.mean(axis=2)
            elif isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values_2d = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                shap_values_2d = shap_values

            # 确保行数匹配
            min_samples = min(len(shap_values_2d), len(X_sample_df))
            shap_values_final = shap_values_2d[:min_samples]
            X_test_final = X_sample_df.iloc[:min_samples]

            # --- 1. Summary Plot ---
            try:
                plt.figure()
                shap.summary_plot(shap_values_final, X_test_final, show=False)
                plt.title(f'SHAP Summary Plot\n({task} - {fill_method} - {model_name})')
                plt.savefig(os.path.join(output_dir, 'summary_plot.png'), dpi=OUTPUT_CONFIG['image_dpi'], bbox_inches='tight')
                plt.close()
                print(f"         - Summary Plot 已保存。")
            except Exception as e:
                print(f"         [警告] Summary Plot生成失败: {e}")

            # --- 2. Dependence Plots ---
            try:
                print(f"         生成 {SHAP_CONFIG['top_features_dependence']} 个Dependence Plots...")
                # 修复特征重要性计算 - 添加维度检查
                if len(shap_values_final.shape) > 1:
                    feature_importance = np.abs(shap_values_final).mean(0)
                else:
                    feature_importance = np.abs(shap_values_final)

                # 确保有足够的特征进行分析
                if len(feature_importance) >= SHAP_CONFIG['top_features_dependence']:
                    top_indices = [int(i) for i in np.argsort(feature_importance)[-SHAP_CONFIG['top_features_dependence']:]]

                    for i, feature_idx in enumerate(top_indices):
                        try:
                            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature_{feature_idx}'
                            plt.figure(figsize=(8, 6))
                            # 修复调用方式：使用特征索引而不是特征名
                            shap.dependence_plot(
                                feature_idx,  # 使用索引而不是特征名
                                shap_values_final,
                                X_test_final,
                                feature_names=feature_names,
                                show=False
                            )
                            plt.title(f'SHAP Dependence Plot for "{feature_name}"')
                            plt.savefig(os.path.join(output_dir, f'dependence_plot_{i+1}_{feature_name}.png'), dpi=OUTPUT_CONFIG['image_dpi'], bbox_inches='tight')
                            plt.close()
                        except Exception as dep_e:
                            print(f"         [警告] Dependence Plot {i+1} 生成失败: {dep_e}")
                            plt.close()
                            continue
                else:
                    print(f"         特征数量不足，跳过Dependence Plots")
            except Exception as e:
                print(f"         [警告] Dependence Plots生成失败: {e}")

            # --- 3. Force Plots ---
            try:
                print(f"         生成 {SHAP_CONFIG['n_force_plot_samples']} 个Force Plots...")
                if hasattr(explainer, 'expected_value'):
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (np.ndarray, list)):
                        expected_value = expected_value[0] if len(expected_value) > 0 else 0
                    elif not isinstance(expected_value, (int, float)):
                        expected_value = float(expected_value)

                    # 确保使用匹配维度的数据
                    n_samples = min(SHAP_CONFIG['n_force_plot_samples'], len(shap_values_final))

                    force_plots_created = 0
                    for i in range(n_samples):
                        try:
                            # 确保SHAP值是1维的
                            shap_values_sample = shap_values_final[i]
                            if len(shap_values_sample.shape) > 1:
                                shap_values_sample = shap_values_sample.flatten()

                            # 获取样本特征值
                            sample_data = X_test_final.iloc[i]

                            success = self._create_custom_force_plot(
                                expected_value,
                                shap_values_sample,
                                sample_data,
                                feature_names,
                                output_dir,
                                f'force_plot_sample_{sample_data.name}.png',
                                task
                            )

                            if success:
                                force_plots_created += 1

                        except Exception as force_e:
                            print(f"         [警告] Force Plot {i+1} 生成失败: {force_e}")
                            continue

                    if force_plots_created > 0:
                        print(f"         成功生成 {force_plots_created} 个Force Plots")
                    else:
                        print(f"         所有Force Plots都生成失败")
                else:
                    print("         [警告] Explainer没有expected_value属性，无法生成Force Plots。")
            except Exception as e:
                print(f"         [警告] Force Plots生成失败: {e}")

        except Exception as e:
            import traceback
            print(f"      [错误] SHAP值计算或绘图失败: {e}")
            traceback.print_exc()

    def _create_custom_force_plot(self, expected_value, shap_values_sample, sample_data, feature_names,
                                  output_dir, filename, task_name):
        """创建自定义的、静态的Force Plot条形图。"""
        try:
            # 确保expected_value是标量
            expected_value_scalar = float(expected_value.item() if isinstance(expected_value, np.ndarray) else expected_value)

            # 确保SHAP值是numpy数组
            shap_values_clean = np.array(shap_values_sample, dtype=np.float64)
            if len(shap_values_clean.shape) > 1:
                shap_values_clean = shap_values_clean.flatten()

            # 处理sample_data - 支持pandas Series和numpy array
            if hasattr(sample_data, 'values'):
                sample_data_clean = np.array(sample_data.values, dtype=np.float64)
            else:
                sample_data_clean = np.array(sample_data, dtype=np.float64)

            prediction = expected_value_scalar + np.sum(shap_values_clean)

            abs_shap_values = np.abs(shap_values_clean)
            sorted_indices = np.argsort(abs_shap_values)[::-1]

            n_features_to_show = min(15, len(shap_values_clean))
            top_indices = [int(i) for i in sorted_indices[:n_features_to_show]]

            top_shap_values = shap_values_clean[top_indices]
            top_feature_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
            top_feature_values = [sample_data_clean[i] if i < len(sample_data_clean) else 0.0 for i in top_indices]

            fig, ax = plt.subplots(figsize=(14, max(6, n_features_to_show * 0.45)))
            y_pos = np.arange(len(top_shap_values))
            colors = ['#FF6B6B' if val > 0 else '#4ECDC4' for val in top_shap_values]

            bars = ax.barh(y_pos, top_shap_values, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

            labels = [f'{name}\n= {val:.3f}' for name, val in zip(top_feature_names, top_feature_values)]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()

            ax.set_xlabel('SHAP Value (对模型输出的影响)', fontsize=10, fontweight='bold')
            ax.set_title(f'SHAP Force Plot - {task_name}\n' +
                         f'基线值: {expected_value_scalar:.3f} -> 模型预测输出: {prediction:.3f}',
                         fontsize=14, fontweight='bold', pad=20)

            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            sns.despine(left=True, bottom=True)

            plt.tight_layout(pad=1.5)

            plt.savefig(os.path.join(output_dir, filename), dpi=OUTPUT_CONFIG['image_dpi'], facecolor='white')
            plt.close(fig)
            return True

        except Exception as e:
            print(f"         [警告] 自定义Force Plot生成失败: {e}")
            plt.close()
            return False

    def _make_predictions(self, model, X_test_df: pd.DataFrame, y_test: pd.Series):
        print("      进行预测...")
        if DEEP_LEARNING_AVAILABLE and isinstance(model, torch.nn.Module):
            model.eval()
            model_device = next(model.parameters()).device
            with torch.no_grad():
                test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32).to(model_device)
                outputs = model(test_tensor)
                y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = model.predict(X_test_df)
            y_proba = model.predict_proba(X_test_df)

        confidence = np.max(y_proba, axis=1)

        results_df = pd.DataFrame({
            '真实标签': y_test.values, '预测标签': y_pred,
            '置信度': confidence, '预测正确': y_test.values == y_pred
        })
        for i in range(y_proba.shape[1]):
            results_df[f'类别_{i}_概率'] = y_proba[:, i]

        return results_df

    def run_analysis_pipeline(self):
        if not self.best_combinations:
            print("没有找到可分析的最佳模型组合，脚本终止。")
            return

        test_df, preprocessors = self._load_data_and_preprocessors()

        all_prediction_results = {}

        for combo in tqdm(self.best_combinations, desc="分析最佳模型"):
            task = combo['任务']

            if task not in test_df.columns or test_df[task].isnull().all():
                print(f"\n跳过任务'{task}'，因为它在测试集中不存在或全为空值。")
                continue

            print(f"\n--- 分析组合: {task} | {combo['填充方法']} | {combo['模型']} ---")

            task_test_df = test_df.dropna(subset=[task]).copy()
            y_test = task_test_df[task].astype(int)

            pca_feature_cols = [col for col in task_test_df.columns if col.startswith('PCA_')]

            if not pca_feature_cols:
                print(f"      [错误] 在测试数据中找不到任何PCA特征列 (以'PCA_'开头)，跳过此组合。")
                continue

            X_test_df = task_test_df[pca_feature_cols]

            num_features = len(pca_feature_cols)
            num_classes = len(np.unique(y_test)) if len(np.unique(y_test)) > 1 else 2

            try:
                model = self._load_best_model(combo, num_features, num_classes)
                if model is None: continue

                self._generate_shap_visualizations(model, combo['模型'], X_test_df, combo)

                prediction_df = self._make_predictions(model, X_test_df, y_test)

                final_prediction_df = pd.concat([
                    task_test_df[[self.config.DATA_CONFIG['subject_id_col'], self.config.DATA_CONFIG['time_col']]].reset_index(drop=True),
                    prediction_df
                ], axis=1)

                sheet_name = f"{task[:10]}_{combo['填充方法']}_{combo['模型']}"
                all_prediction_results[sheet_name] = final_prediction_df

            except Exception as e:
                import traceback
                print(f"      [错误] 处理组合 {combo} 时失败: {e}")
                traceback.print_exc()

            gc.collect()

        if all_prediction_results:
            self._save_predictions_to_excel(all_prediction_results)

        print("\n" + "="*80)
        print("模型分析与预测流水线执行完毕!")
        print("="*80)

    def _save_predictions_to_excel(self, results_dict: dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"best_models_predictions_{timestamp}.xlsx"
        output_path = os.path.join(self.path_config['predictions_dir'], output_filename)

        print(f"\n正在保存预测报告: {output_path}")
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in results_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            print("预测报告保存成功！")
        except Exception as e:
            print(f"错误：保存预测报告失败 - {e}")

def main():
    try:
        analyzer = ModelAnalyzer()
        analyzer.run_analysis_pipeline()
        print("\n所有分析任务完成。")
    except Exception as e:
        import traceback
        print(f"\n脚本执行过程中发生严重错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
