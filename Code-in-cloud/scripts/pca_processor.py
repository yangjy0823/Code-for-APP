# scripts/csv_pca_processor.py

"""
PCA预处理器脚本
功能:
1. 分别加载由 data_preprocessing.py 生成的 train_data.csv 和 test_data.csv。
2. 仅在训练数据上拟合(fit) StandardScaler 和 PCA。
3. 使用拟合好的转换器来转换(transform)训练集和测试集。
4. 保存PCA处理后的 pca_train_data.csv 和 pca_test_data.csv。
5. 将拟合好的 scaler 和 pca 对象保存为.pkl文件，供后续分析脚本使用。
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- 1. 导入配置 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#
from configuration.config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG

class CSVPCAProcessor:
    def __init__(self):
        """从config模块初始化所有配置"""
        self.data_config = DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.path_config = PATH_CONFIG
        self.pca_params = self.model_config['feature_processing']['pca_params']
        print("初始化PCA处理器 (配置驱动)")

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """
        根据数据类型和配置识别特征列。
        核心改进: 能够处理聚合后带后缀的列名 (如 '性别_first', '疲劳分类标签_first')。
        """
        prefix_exclude_bases = self.data_config['target_columns'] + [self.data_config['gender_col']]
        exact_exclude = {
            self.data_config['subject_id_col'],
            self.data_config['time_col'],
            'time_window'
        }

        feature_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in exact_exclude:
                continue

            is_target_or_gender = False
            for base in prefix_exclude_bases:
                if col.startswith(base):
                    is_target_or_gender = True
                    break

            if is_target_or_gender:
                continue

            feature_cols.append(col)

        return feature_cols

    def run(self):
        """执行完整的PCA处理流程"""
        # --- Phase 1: 处理训练数据 (Fit and Transform) ---
        print("\n--- 阶段1: 处理训练数据 ---")
        train_df = pd.read_csv(self.data_config['train_data_path'])
        print(f"成功加载训练数据: {self.data_config['train_data_path']}，形状: {train_df.shape}")

        feature_cols = self._get_feature_cols(train_df)
        print(f"识别出 {len(feature_cols)} 个特征列。")

        print("-" * 20)
        print("参与PCA的原始特征项列表:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:>2}. {col}")
        print("-" * 20)

        X_train = train_df[feature_cols].fillna(train_df[feature_cols].mean())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print("StandardScaler在训练数据上拟合和转换完成。")

        pca = PCA(n_components=self.pca_params['n_components'], random_state=self.pca_params['random_state'])
        X_train_pca = pca.fit_transform(X_train_scaled)
        print("PCA在训练数据上拟合和转换完成。")
        print(f"原始特征维度: {pca.n_features_in_}, PCA降维后维度: {pca.n_components_}")
        print(f"累计方差解释比: {pca.explained_variance_ratio_.sum():.4f}")

        pca_cols = [f'PCA_PC{i+1}' for i in range(X_train_pca.shape[1])]
        pca_train_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=train_df.index)

        metadata_train_df = train_df.drop(columns=feature_cols)
        final_train_df = pd.concat([metadata_train_df, pca_train_df], axis=1)
        final_train_df.to_csv(self.data_config['pca_train_data_path'], index=False, encoding='utf-8-sig')
        print(f"PCA处理后的训练数据已保存至: {self.data_config['pca_train_data_path']}")

        # --- Phase 2: 保存拟合好的转换器 ---
        print("\n--- 阶段2: 保存Scaler和PCA模型 ---")
        model_path = self.path_config['pca_model_path']
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        pca_model_payload = {
            'scaler': scaler,
            'pca': pca,
            'feature_cols': feature_cols
        }
        with open(self.path_config['pca_model_path'], 'wb') as f:
            pickle.dump(pca_model_payload, f)
        print(f"Scaler和PCA对象已保存至: {self.path_config['pca_model_path']}")

        # --- Phase 3: 处理测试数据 (Transform Only) ---
        print("\n--- 阶段3: 处理测试数据 ---")
        test_df = pd.read_csv(self.data_config['test_data_path'])
        print(f"成功加载测试数据: {self.data_config['test_data_path']}，形状: {test_df.shape}")

        X_test = test_df[feature_cols].fillna(train_df[feature_cols].mean())

        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        print("使用已有的Scaler和PCA转换测试数据完成。")

        pca_test_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=test_df.index)
        metadata_test_df = test_df.drop(columns=feature_cols)
        final_test_df = pd.concat([metadata_test_df, pca_test_df], axis=1)
        final_test_df.to_csv(self.data_config['pca_test_data_path'], index=False, encoding='utf-8-sig')
        print(f"PCA处理后的测试数据已保存至: {self.data_config['pca_test_data_path']}")


# --- 新增的 main 函数 ---
def main():
    """
    脚本的主执行函数，封装了所有操作。
    """
    print("="*60)
    print("启动PCA预处理流程")
    print("="*60)

    try:
        processor = CSVPCAProcessor()
        processor.run()

        print("\n" + "="*60)
        print("PCA预处理流程成功完成!")
        print("下一步: python scripts/train_models.py")
        print("="*60)

    except Exception as e:
        import traceback
        print(f"\nPCA处理过程中发生严重错误: {e}")
        traceback.print_exc()

# --- 修改后的 __name__ == "__main__" 块 ---
if __name__ == "__main__":
    main()
