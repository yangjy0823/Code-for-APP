# predict_external_data.py

"""
外部数据预测器 - 直接在原文件的分类标签列填入预测结果
使用方法: python predict_external_data.py path/to/your/external_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle

# 添加项目路径configs
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from configuration import config
from scripts import utils
from scripts.data_preprocessing import ComprehensiveDataPreprocessor
from models.deep_learning import LSTMClassifier, GRUClassifier, CNNLSTMClassifier

try:
    import torch
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


def predict_and_fill(data_path):
    """预测并填充原数据的分类标签列"""
    print(f"处理文件: {os.path.basename(data_path)}")

    # 1. 加载原始数据
    original_df = pd.read_csv(data_path, encoding='gbk') if data_path.endswith('.csv') else pd.read_excel(data_path)
    print(f"原始数据形状: {original_df.shape}")

    # 2. 预处理数据并保持映射关系
    best_models = load_best_models()
    processed_df, mapping_df = preprocess_with_mapping(original_df)

    # 3. 获取PCA特征并进行预测
    pca_cols = [col for col in processed_df.columns if col.startswith('PCA_')]
    X_features = processed_df[pca_cols]

    for model_info in best_models:
        task = model_info['任务']
        if task not in original_df.columns:
            continue

        try:
            model = load_model(model_info, len(pca_cols))
            if model is None:
                continue

            predictions = predict_single_task(model, X_features, model_info['模型'])

            # 将聚合层级的预测结果映射回原始数据
            original_df[task] = map_predictions_to_original(predictions, mapping_df, original_df)
            print(f"✓ {task} 预测完成")

        except Exception as e:
            print(f"✗ {task} 预测失败: {e}")

    # 4. 保存结果
    output_path = data_path.replace('.csv', '_predicted.csv').replace('.xlsx', '_predicted.xlsx')
    if data_path.endswith('.csv'):
        original_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        original_df.to_excel(output_path, index=False)

    print(f"预测完成! 结果已保存至: {output_path}")
    return output_path


def load_best_models():
    """加载最佳模型信息"""
    try:
        report_file = utils.find_latest_file(config.PATH_CONFIG['evaluation_reports_dir'], ".xlsx")
        df = pd.read_excel(report_file, sheet_name='总体最佳模型对比')
        return [{'任务': row['任务'], '填充方法': row['填充方法'], '模型': row['模型'].replace('_', '-')}
                for _, row in df.iterrows()]
    except:
        return [{'任务': task, '填充方法': 'forward', '模型': 'random-forest'}
                for task in config.TARGET_VARIABLES]


def load_pca_preprocessors():
    """加载PCA预处理器"""
    with open(config.PATH_CONFIG['pca_model_path'], 'rb') as f:
        return pickle.load(f)


def preprocess_with_mapping(df):
    """预处理数据并保持原始数据到聚合数据的映射关系"""
    processor = ComprehensiveDataPreprocessor()

    # 1. 清洗数据并创建映射索引
    df_clean = processor.clean_and_standardize_data(df)
    df_clean['_original_index'] = df_clean.index

    # 2. 创建时间窗口
    df_windowed = processor.create_time_windows(df_clean)
    df_filled = processor.handle_missing_values(df_windowed)

    # 3. 创建映射关系 (聚合后每行对应的原始数据索引范围)
    subject_col = config.DATA_CONFIG['subject_id_col']
    time_col = config.DATA_CONFIG['time_col']

    mapping_list = []
    for _, row in df_filled.iterrows():
        subject_id = row[subject_col]
        window_time = row[time_col]

        # 查找对应的原始数据索引
        mask = (
            (df_clean[subject_col] == subject_id) &
            (df_clean[time_col].dt.floor('1min') == window_time)
        )
        original_indices = df_clean.loc[mask, '_original_index'].tolist()
        mapping_list.append(original_indices)

    mapping_df = pd.DataFrame({'original_indices': mapping_list})

    # 4. 应用PCA转换
    processed_df = apply_pca_transform(df_filled)

    return processed_df, mapping_df


def apply_pca_transform(df):
    """应用PCA转换"""
    preprocessors = load_pca_preprocessors()
    exclude_cols = {config.DATA_CONFIG['subject_id_col'], config.DATA_CONFIG['time_col'],
                   config.DATA_CONFIG['gender_col'], '_original_index'}
    exclude_cols.update(config.TARGET_VARIABLES)

    feature_cols = [col for col in df.columns
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    expected_features = preprocessors['feature_cols']
    available_features = [col for col in expected_features if col in feature_cols]

    X = df[available_features].fillna(df[available_features].mean())
    X_scaled = preprocessors['scaler'].transform(X)
    X_pca = preprocessors['pca'].transform(X_scaled)

    pca_cols = [f'PCA_PC{i+1}' for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    metadata_cols = [col for col in df.columns if col in exclude_cols and col in df.columns]
    return pd.concat([df[metadata_cols], pca_df], axis=1)


def map_predictions_to_original(predictions, mapping_df, original_df):
    """将聚合层级的预测结果映射回原始数据"""
    result = pd.Series(index=original_df.index, dtype=int)

    for i, pred in enumerate(predictions):
        original_indices = mapping_df.iloc[i]['original_indices']
        for orig_idx in original_indices:
            if orig_idx in result.index:
                result.loc[orig_idx] = pred

    return result.fillna(0)


def load_model(model_info, num_features):
    """加载模型"""
    task, fill_method, model_name = model_info['任务'], model_info['填充方法'], model_info['模型']

    ext = ".pth" if model_name in ['lstm', 'gru', 'cnn-lstm'] else ".pkl"
    model_file = f"{task}_{fill_method}_{model_name}{ext}"
    model_path = os.path.join(config.PATH_CONFIG['models_dir'], model_file)

    if not os.path.exists(model_path):
        return None

    if ext == ".pth" and DEEP_LEARNING_AVAILABLE:
        model_instance = create_pytorch_model(model_name, num_features, 2)
        return utils.load_model(model_path, model_instance=model_instance)
    else:
        return utils.load_model(model_path)


def create_pytorch_model(model_name, num_features, num_classes):
    """创建PyTorch模型实例"""
    params = config.MODEL_CONFIG['deep_models'][model_name.replace('-', '_')]

    if model_name == 'lstm':
        return LSTMClassifier(
            input_size=num_features, num_classes=num_classes,
            hidden_size=params['hidden_size'][0], num_layers=params['num_layers'][0],
            dropout=params['dropout'][0], bidirectional=params['bidirectional'][0]
        )
    elif model_name == 'gru':
        return GRUClassifier(
            input_size=num_features, num_classes=num_classes,
            hidden_size=params['hidden_size'][0], num_layers=params['num_layers'][0],
            dropout=params['dropout'][0], bidirectional=params['bidirectional'][0]
        )
    elif model_name == 'cnn-lstm':
        return CNNLSTMClassifier(
            input_size=num_features, num_classes=num_classes, dropout=params['dropout'][0],
            cnn_filters=params['cnn_filters'][0], kernel_size=params['kernel_sizes'][0],
            lstm_hidden_size=params['lstm_hidden_size'][0], lstm_num_layers=params['lstm_num_layers'][0]
        )


def predict_single_task(model, X_features, model_name):
    """执行单个任务预测"""
    if DEEP_LEARNING_AVAILABLE and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_features.values, dtype=torch.float32)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)
            outputs = model(X_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            return np.argmax(proba, axis=1)
    else:
        return model.predict(X_features)


def main(data_path=None):
    """主函数"""
    if data_path is None:
        if len(sys.argv) < 2:
            print("使用方法: python predict_external_data.py <数据文件路径>")
            return
        data_path = sys.argv[1]

    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return

    try:
        predict_and_fill(data_path)
    except Exception as e:
        print(f"预测过程出错: {e}")


if __name__ == "__main__":
    main('test_data.xlsx')
