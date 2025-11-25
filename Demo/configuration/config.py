# configs/config.py

import os
import torch

# --- 1. 基础路径定义 ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- 2. 统一路径配置 (PATH_CONFIG) ---
# 经过重构，路径更加清晰，严格区分了原始数据、处理后、拆分后和最终产出
PATH_CONFIG = {
    # 原始数据
    "raw_data_dir": os.path.join(BASE_DIR, "data/raw"),
    
    # 预处理和特征工程后的数据存放目录
    "processed_data_dir": os.path.join(BASE_DIR, "data/processed"),

    # --- 核心变更: 训练集/测试集拆分路径 ---
    "splits_dir": os.path.join(BASE_DIR, "data/splits"),
    "train_data_path": os.path.join(BASE_DIR, "data/splits/train_data.csv"),
    "test_data_path": os.path.join(BASE_DIR, "data/splits/test_data.csv"),
    "pca_train_data_path": os.path.join(BASE_DIR, "data/splits/pca_train_data.csv"),
    "pca_test_data_path": os.path.join(BASE_DIR, "data/splits/pca_test_data.csv"),
    
    # --- 核心变更: PCA/Scaler模型保存路径 ---
    "pca_model_path": os.path.join(BASE_DIR, "results/models/pca_scaler_models.pkl"),

    # 模型输出路径
    "models_dir": os.path.join(BASE_DIR, "results/models"),
    "checkpoints_dir": os.path.join(BASE_DIR, "results/checkpoints"),

    # 结果与报告输出路径
    "plots_dir": os.path.join(BASE_DIR, "results/plots"),
    "evaluation_reports_dir": os.path.join(BASE_DIR, "results/evaluation_reports"),
    "shap_analysis_dir": os.path.join(BASE_DIR, "results/shap_analysis"),
    "predictions_dir": os.path.join(BASE_DIR, "results/predictions")
}

# --- 3. 全局常量定义 ---
WINDOW_SIZE_MINUTES = 1
SEQUENCE_LENGTH = 15
TEST_SUBJECTS = [1, 2]  # 用于数据拆分的测试集受试者ID

# 目标变量
TARGET_VARIABLES = [
    "疲劳分类标签", "血糖分类标签", "水合状态分类标签", 
    "乳酸分类标签", "肌肉疲劳分类标签", "蛋白供应分类标签"
]

# 评估指标
EVALUATION_METRICS = [
    "f1_macro", "f1_weighted", "precision_macro", 
    "recall_macro", "roc_auc_ovr", "accuracy"
]

# 特征列定义 (单一数据源)
HIGH_FREQ_FEATURES = ["Na (mM)", "K (mM)", "Glucose (uM)", "Lactate (mM)", "SUN (mM)"]
MID_FREQ_FEATURES = ["MDF", "MEF"]
LOW_FREQ_FEATURES = ["Cortisol (μg/L)", "Tes(ng/L)"]

# 注意：此列表主要用于train_models.py中，处理因时序特征工程（如lag）产生的NaN
FILLING_STRATEGIES = ["linear", "forward", "backward"]


# --- 4. 模块化配置对象 ---

# 数据配置 (DATA_CONFIG)
DATA_CONFIG = {
    # 路径配置
    'raw_data_dir': PATH_CONFIG['raw_data_dir'],
    'processed_data_dir': PATH_CONFIG['processed_data_dir'],
    'train_data_path': PATH_CONFIG['train_data_path'],
    'test_data_path': PATH_CONFIG['test_data_path'],
    'pca_train_data_path': PATH_CONFIG['pca_train_data_path'],
    'pca_test_data_path': PATH_CONFIG['pca_test_data_path'],

    # 数据列名
    'subject_id_col': 'Subject',
    'time_col': 'Time', 
    'gender_col': '性别',
    'target_columns': TARGET_VARIABLES,
    
    # 数据处理参数
    'window_size': WINDOW_SIZE_MINUTES,
    'sequence_length': SEQUENCE_LENGTH,
    
    # 特征分组 (直接引用上方定义的列表)
    'high_freq_features': HIGH_FREQ_FEATURES,
    'medium_freq_features': MID_FREQ_FEATURES,
    'low_freq_features': LOW_FREQ_FEATURES,
}

# 模型配置 (MODEL_CONFIG)
MODEL_CONFIG = {
    'random_state': 42,
    'test_subjects': TEST_SUBJECTS,
    'cv_folds': 5,
    
    'tree_models': {
        'random_forest': { 'n_estimators': [100, 200], 'max_depth': [10, 20] },
        'lightgbm': { 'num_leaves': [31, 50], 'learning_rate': [0.05, 0.1] }
    },
    
    # 深度学习模型参数 (核心变更: 增加 input_size 占位符)
    'deep_models': {
        'lstm': {
            'input_size': None,  # 将在训练脚本中根据PCA结果动态填充
            'hidden_size': [64, 128], 'num_layers': [1, 2], 'dropout': [0.2, 0.3],
            'bidirectional': [False, True], 'batch_size': [32, 64], 'epochs': 50,
            'learning_rate': [0.001, 0.01], 'weight_decay': [0.0, 0.0001]
        },
        'gru': {
            'input_size': None,
            'hidden_size': [64, 128], 'num_layers': [1, 2], 'dropout': [0.2, 0.3],
            'bidirectional': [False, True], 'batch_size': [32, 64], 'epochs': 50,
            'learning_rate': [0.001, 0.01], 'weight_decay': [0.0, 0.0001]
        },
        'cnn_lstm': {
            'input_size': None,
            'cnn_filters': [32, 64], 'kernel_sizes': [3, 5],
            'lstm_hidden_size': [64, 128], 'lstm_num_layers': [1, 2],
            'dropout': [0.2, 0.3], 'batch_size': [32, 64], 'epochs': 50,
            'learning_rate': [0.001, 0.01], 'weight_decay': [0.0, 0.0001]
        }
    },
    
    'pytorch_config': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0, 'pin_memory': False, 'early_stopping_patience': 10,
        'lr_scheduler': { 'type': 'ReduceLROnPlateau', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-5 },
        'gradient_clipping': { 'enabled': True, 'max_norm': 1.0 }
    },
    
    'feature_processing': {
        'apply_pca': True,
        'pca_params': { 'n_components': 0.95, 'random_state': 42, 'min_features_for_pca': 20 }
    }
}

# 评估配置 (EVALUATION_CONFIG)
EVALUATION_CONFIG = {
    'metrics': EVALUATION_METRICS,
    'cv_scoring': 'f1_weighted',
    'class_balance': True,
    'plot_results': True,
    'save_predictions': True,
    'generate_reports': True
}