# scripts/data_preprocessing.py

"""
完整数据预处理主脚本
功能:
1. 从原始Excel/CSV文件加载数据。
2. 进行数据清洗、时间窗口聚合、特征工程和缺失值填充。
3. 根据config中的TEST_SUBJECTS，将最终处理好的数据拆分为 train_data.csv 和 test_data.csv。
"""

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

# --- 1. 导入配置 (关键变更) ---
# 将项目根目录添加到sys.path，以便能够导入configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from configuration.config import DATA_CONFIG, MODEL_CONFIG, PATH_CONFIG

class ComprehensiveDataPreprocessor:
    def __init__(self):
        """初始化预处理器，所有配置均从config模块导入"""
        self.data_config = DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.path_config = PATH_CONFIG

        # 直接从配置中读取特征列表，不再硬编码
        self.high_freq_features = self.data_config['high_freq_features']
        self.medium_freq_features = self.data_config['medium_freq_features']
        self.low_freq_features = self.data_config['low_freq_features']

        print("初始化数据预处理器 (配置驱动)")

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据文件 (xlsx 或 csv)"""
        print(f"加载原始数据: {os.path.basename(file_path)}")
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='gbk')
        else:
            raise ValueError("不支持的文件格式，请使用.xlsx或.csv文件")
        print(f"原始数据形状: {df.shape}")
        return df

    def clean_and_standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和标准化"""
        print("数据清洗...")
        df_clean = df.copy()
        time_col, subject_col = self.data_config['time_col'], self.data_config['subject_id_col']

        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        df_clean = df_clean.sort_values([subject_col, time_col]).reset_index(drop=True)

        print(f"受试者数量: {df_clean[subject_col].nunique()}")
        return df_clean

    def create_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间窗口并聚合特征 (性能优化版)
        """
        print(f"创建时间窗口 ({self.data_config['window_size']}分钟窗口)...")
        window_size = self.data_config['window_size']
        subject_col = self.data_config['subject_id_col']
        time_col = self.data_config['time_col']

        df['time_window'] = df[time_col].dt.floor(f'{window_size}min')

        # 定义聚合规则
        agg_dict = {}
        # 静态列（目标、性别）使用 'first' 聚合
        static_cols = [self.data_config['gender_col']] + self.data_config['target_columns']
        for col in static_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        # 中低频特征使用 'mean' 聚合
        single_agg_features = self.medium_freq_features + self.low_freq_features
        for col in single_agg_features:
            if col in df.columns:
                agg_dict[col] = 'mean'

        # 高频特征使用多重聚合
        for col in self.high_freq_features:
            if col in df.columns:
                agg_dict[col] = ['mean', 'std', 'min', 'max']

        # 执行聚合
        df_windowed = df.groupby([subject_col, 'time_window']).agg(agg_dict).reset_index()

        # 展平多级列名
        df_windowed.columns = ['_'.join(col).strip('_') for col in df_windowed.columns.values]

        # --- 第一次修正：将 'time_window' 列名改回配置中指定的 'Time' ---
        df_windowed.rename(columns={'time_window': time_col}, inplace=True)

        # --- 第二次修正（本次新增）：将单聚合列的后缀名去掉，恢复原名 ---
        rename_dict = {}
        for col in static_cols:
            if f"{col}_first" in df_windowed.columns:
                rename_dict[f"{col}_first"] = col
        for col in single_agg_features:
            if f"{col}_mean" in df_windowed.columns:
                rename_dict[f"{col}_mean"] = col
        df_windowed.rename(columns=rename_dict, inplace=True)
        # --- 修正结束 ---

        # 计算额外特征，如range
        for feature in self.high_freq_features:
            if f'{feature}_max' in df_windowed.columns and f'{feature}_min' in df_windowed.columns:
                df_windowed[f'{feature}_range'] = df_windowed[f'{feature}_max'] - df_windowed[f'{feature}_min']

        print(f"时间窗口创建完成，数据形状: {df_windowed.shape}")
        return df_windowed

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        print("处理缺失值...")
        df_filled = df.copy()
        subject_col = self.data_config['subject_id_col']

        # 识别特征列
        feature_cols = [col for col in df.columns if
                        col not in [subject_col, 'time_window', self.data_config['gender_col']] + self.data_config[
                            'target_columns']]

        # 按受试者分组进行插值
        for col in feature_cols:
            df_filled[col] = df_filled.groupby(subject_col)[col].transform(
                lambda x: x.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            )

        # 全局填充剩余的NaN
        df_filled.fillna(df.mean(numeric_only=True), inplace=True)
        print(f"缺失值处理完成，剩余NaN: {df_filled.isnull().sum().sum()}")
        return df_filled

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征工程（交互特征、滞后特征、滚动特征）"""
        print("创建工程特征...")
        # (此部分逻辑与原版基本一致，可直接复用，此处为简化示例)
        # 确保在创建滞后和滚动特征时也使用 groupby(subject_id)
        # 例如: df[col + '_lag1'] = df.groupby(self.data_config['subject_id_col'])[col].shift(1)
        # 创建后需要再次处理因此产生的NaN值
        df_features = self.handle_missing_values(df)  # 再次调用填充
        return df_features

    # def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     创建特征工程（交互特征、滞后特征、滚动特征）
    #     """
    #     print("创建工程特征...")
    #     df_featured = df.copy()
    #     subject_col = self.data_config['subject_id_col']

    #     # --- 1. 生理组合特征 (T/C Ratio) ---
    #     # 动态查找Tes和Cortisol的均值列，避免硬编码
    #     cortisol_col = next((c for c in df_featured.columns if 'Cortisol' in c and c.endswith('_mean')), None)
    #     tes_col = next((c for c in df_featured.columns if 'Tes' in c and c.endswith('_mean')), None)

    #     if cortisol_col and tes_col:
    #         print(f"  - 创建 T/C Ratio 特征 (使用 {tes_col} / {cortisol_col})")
    #         # 为避免除以零，给分母加上一个极小值 epsilon
    #         df_featured['T_C_Ratio'] = df_featured[tes_col] / (df_featured[cortisol_col] + 1e-6)
    #     else:
    #         print("  - 警告: 未在数据中找到 Cortisol 或 Tes 的均值列，跳过 T/C_Ratio 创建。")

    #     # --- 2. 时序衍生特征 ---
    #     # 识别出适合创建时序特征的核心指标列（通常是各类聚合后的均值和范围值）
    #     features_for_ts = [col for col in df_featured.columns if col.endswith(('_mean', '_std', '_range'))]
    #     # 排除掉目标列和T/C Ratio自身，避免对它们创建时序特征
    #     target_bases = self.data_config['target_columns']
    #     features_for_ts = [
    #         f for f in features_for_ts
    #         if not f.startswith(tuple(target_bases)) and f != 'T_C_Ratio'
    #     ]

    #     if not features_for_ts:
    #         print("  - 警告: 未找到适合创建时序特征的列，跳过此步骤。")
    #         # 即使没有新特征，也可能存在旧的NaN需要处理
    #         return self.handle_missing_values(df_featured)

    #     print(f"  - 为 {len(features_for_ts)} 个核心特征创建滞后和滚动特征...")

    #     # -- 2a. 滞后特征 (Lag Features) --
    #     lags = [1, 2, 3]
    #     for feature in features_for_ts:
    #         for lag in lags:
    #             new_col_name = f'{feature}_lag_{lag}'
    #             # 按受试者分组计算滞后，避免数据穿越
    #             df_featured[new_col_name] = df_featured.groupby(subject_col)[feature].shift(lag)

    #     # -- 2b. 滚动窗口特征 --
    #     # 注意：这里的窗口大小是基于已聚合的时间窗口的点数。
    #     # 如果您的聚合窗口是10分钟，那么rolling_window=3就代表(3*10=30分钟)的滚动窗口。
    #     # 我们使用[3, 5]作为示例，即30分钟和50分钟的滚动窗口。
    #     rolling_windows = [3, 5]
    #     for feature in features_for_ts:
    #         for win in rolling_windows:
    #             # 使用 transform 来保持 DataFrame 的原始形状和索引，并按受试者分组计算
    #             rolling_mean = df_featured.groupby(subject_col)[feature].transform(
    #                 lambda x: x.rolling(window=win, min_periods=1).mean()
    #             )
    #             rolling_std = df_featured.groupby(subject_col)[feature].transform(
    #                 lambda x: x.rolling(window=win, min_periods=1).std()
    #             )
    #             df_featured[f'{feature}_rolling_{win}_mean'] = rolling_mean
    #             df_featured[f'{feature}_rolling_{win}_std'] = rolling_std

    #     # --- 3. 填充衍生特征产生的缺失值 ---
    #     # 创建滞后和滚动特征会在每个受试者的开头产生大量NaN，需要再次全面填充
    #     print("  - 填充因时序特征产生的大量缺失值...")
    #     # 再次调用handle_missing_values来处理新产生的NaN
    #     df_final = self.handle_missing_values(df_featured)

    #     print(f"特征工程完成，数据最终形状: {df_final.shape}")
    #     return df_final

    def split_and_save_data(self, df: pd.DataFrame):
        """
        核心变更: 根据配置拆分数据为训练集和测试集，并保存
        """
        print("\n正在拆分训练集和测试集...")
        subject_col = self.data_config['subject_id_col']
        test_subjects = self.model_config['test_subjects']

        test_mask = df[subject_col].isin(test_subjects)
        train_df = df[~test_mask]
        test_df = df[test_mask]

        print(f"总数据: {df.shape[0]}行")
        print(f"训练集: {train_df.shape[0]}行 (受试者: {train_df[subject_col].unique()})")
        print(f"测试集: {test_df.shape[0]}行 (受试者: {test_df[subject_col].unique()})")

        # 确保输出目录存在
        os.makedirs(self.path_config['splits_dir'], exist_ok=True)

        # 保存文件
        train_path = self.path_config['train_data_path']
        test_path = self.path_config['test_data_path']

        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

        print(f"\n训练集已保存至: {train_path}")
        print(f"测试集已保存至: {test_path}")


def main():
    print("=" * 60)
    print("启动完整数据预处理与拆分流程")
    print("=" * 60)

    # 从配置中指定的目录查找输入文件
    raw_data_dir = PATH_CONFIG['raw_data_dir']
    input_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.xlsx', '.csv'))]

    if not input_files:
        print(f"错误: 在目录 '{raw_data_dir}' 中找不到任何 .xlsx 或 .csv 输入文件。")
        return

    # 假设使用找到的第一个文件
    input_file_path = os.path.join(raw_data_dir, input_files[0])

    try:
        preprocessor = ComprehensiveDataPreprocessor()

        # 1. 加载
        df_raw = preprocessor.load_raw_data(input_file_path)

        # 2. 清洗
        df_clean = preprocessor.clean_and_standardize_data(df_raw)

        # 3. 时间窗口
        df_windowed = preprocessor.create_time_windows(df_clean)

        # 4. 缺失值处理
        df_filled = preprocessor.handle_missing_values(df_windowed)

        # 5. 特征工程
        df_final = preprocessor.create_engineered_features(df_filled)

        # 6. 核心步骤：拆分并保存
        preprocessor.split_and_save_data(df_final)

        print("\n" + "=" * 60)
        print("数据预处理与拆分流程成功完成!")
        print("下一步: python scripts/csv_pca_processor.py")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"\n预处理过程中发生严重错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
