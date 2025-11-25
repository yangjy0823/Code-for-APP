# 多维度运动员疲劳监测系统

基于机器学习的多模态生理数据分析系统，监测运动员6个关键状态：疲劳、血糖、水合、乳酸、肌肉疲劳、蛋白供应。

## 核心特性

- **一键执行**：完全自动化的数据预处理到模型部署流程
- **18个预测模型**：树模型（RandomForest、LightGBM）+ 深度学习（LSTM、GRU、CNN-LSTM）
- **SHAP可解释性**：自定义Force Plot，126个可视化图表
- **ONNX部署**：跨平台模型导出支持

## 快速开始

### 环境安装
```bash
# 安装所有依赖（包含ONNX功能）
pip install -r requirements.txt
```

### 运行流程
```bash
# 一键执行完整流程（推荐）
python main.py

# 分步执行
python scripts/data_preprocessing.py       # 1. 数据预处理
python scripts/pca_processor.py           # 2. PCA降维  
python scripts/train_models.py            # 3. 模型训练
python scripts/analyze_models.py          # 4. SHAP分析
python scripts/export_models_to_onnx.py   # 5. ONNX导出
```

## 项目结构
```
├── main.py                    # 主执行入口
├── configs/config.py          # 配置文件
├── data/raw/                  # 原始数据
├── scripts/                   # 核心脚本
│   ├── data_preprocessing.py  # 数据预处理
│   ├── pca_processor.py       # PCA降维
│   ├── train_models.py        # 模型训练
│   ├── analyze_models.py      # SHAP分析
│   └── export_models_to_onnx.py # ONNX导出
├── models/deep_learning.py    # 深度学习模型
└── results/                   # 输出结果
```

## 输出文件

- **SHAP可视化**：`results/shap/` - 126个PNG图片
- **预测结果**：`results/predictions/` - 18个Excel工作表
- **ONNX模型**：`results/models/onnx/` - 18个.onnx文件

## 技术亮点

### 数据处理
- **异构时序对齐**：1分钟窗口统一不同采样频率
- **特征工程**：生理组合特征 + 时序衍生特征 + PCA降维
- **填充策略**：ffill、bfill、linear interpolation

### 模型评估
- **GroupKFold**：按受试者分组，防止数据泄露
- **多指标评估**：F1-Score、AUC、Precision、Recall
- **类别平衡**：应对不均衡数据集

### 可解释性
- **全局重要性**：Summary Plot
- **特征依赖**：Dependence Plot  
- **个体解释**：自定义Force Plot（解决SHAP 0.45.0兼容性）

## 系统要求

- Python 3.9+
- 内存：建议8GB+
- 存储：约2GB（包含数据和结果）
- GPU：可选（深度学习加速）

## 版本信息

**版本**：2.0 | **状态**：生产就绪 | **更新**：2025-08-15