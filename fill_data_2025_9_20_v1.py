import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 读取数据
#########################################################
#1,修改读取文件的路径
#2 修改保存文件的名称
#########################################################
data = pd.read_excel('test_data.xlsx')
#data = pd.read_excel('Subject1-实时监测干预组可穿戴数据.xlsx')
# 数据基本信息分析
print("=" * 50)
print("数据基本信息")
print("=" * 50)
print(f"数据形状: {data.shape}")
print(f"数据列名: {list(data.columns)}")
print("\n缺失值统计:")
missing_info = data.isnull().sum()
missing_percentage = (data.isnull().sum() / len(data)) * 100
missing_df = pd.DataFrame({
    '缺失值数量': missing_info,
    '缺失值比例(%)': missing_percentage
})
print(missing_df[missing_df['缺失值数量'] > 0])

# 分离不同类型的列
time_column = 'Time'
categorical_columns = ['Subject', '性别']
numeric_columns = [col for col in data.columns if col not in [time_column] + categorical_columns]

# 获取各类型中有缺失值的列
time_missing = time_column if data[time_column].isnull().any() else None
categorical_missing = [col for col in categorical_columns if data[col].isnull().any()]
numeric_missing = [col for col in numeric_columns if data[col].isnull().any()]

print(f"\n时间列缺失: {time_missing}")
print(f"分类列缺失: {categorical_missing}")
print(f"数值列缺失: {numeric_missing}")


# 定义各种填充方法
class ImputationMethods:
    @staticmethod
    def fill_all_columns(df, strategy='mean'):
        """填充所有类型的列"""
        df_filled = df.copy()

        # 填充时间列
        if time_missing:
            if strategy in ['forward_fill', 'ffill']:
                df_filled[time_missing] = df_filled[time_missing].fillna(method='ffill')
            elif strategy in ['backward_fill', 'bfill']:
                df_filled[time_missing] = df_filled[time_missing].fillna(method='bfill')
            elif strategy == 'linear':
                # 时间列线性插值需要特殊处理
                non_null_times = df_filled[time_missing].dropna()
                if len(non_null_times) >= 2:
                    # 将时间转换为数字进行插值，然后转回时间戳
                    time_idx = non_null_times.index
                    time_vals = non_null_times.astype(np.int64)
                    f = interp1d(time_idx, time_vals, bounds_error=False, fill_value="extrapolate")

                    # 对缺失值位置进行插值
                    missing_idx = df_filled[df_filled[time_missing].isnull()].index
                    if len(missing_idx) > 0:
                        filled_vals = f(missing_idx)
                        df_filled.loc[missing_idx, time_missing] = pd.to_datetime(filled_vals)
            else:
                # 对于其他策略，使用前后填充
                df_filled[time_missing] = df_filled[time_missing].fillna(method='ffill')
                df_filled[time_missing] = df_filled[time_missing].fillna(method='bfill')

        # 填充分类列
        if categorical_missing:
            for col in categorical_missing:
                if strategy == 'mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df_filled[col].fillna(mode_val[0], inplace=True)
                elif strategy in ['forward_fill', 'ffill']:
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                elif strategy in ['backward_fill', 'bfill']:
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                else:
                    # 对于其他策略，默认使用众数
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df_filled[col].fillna(mode_val[0], inplace=True)

        # 填充数值列
        if numeric_missing:
            for col in numeric_missing:
                if strategy == 'mean':
                    df_filled[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df_filled[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df_filled[col].fillna(mode_val[0], inplace=True)
                elif strategy in ['forward_fill', 'ffill']:
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                elif strategy in ['backward_fill', 'bfill']:
                    df_filled[col] = df_filled[col].fillna(method='bfill')
                elif strategy == 'linear':
                    df_filled[col] = df_filled[col].interpolate(method='linear')
                elif strategy == 'polynomial':
                    df_filled[col] = df_filled[col].interpolate(method='polynomial', order=2)
                elif strategy == 'spline':
                    df_filled[col] = df_filled[col].interpolate(method='spline', order=3)
                else:
                    # 默认使用均值
                    df_filled[col].fillna(df[col].mean(), inplace=True)

            # 处理首尾仍然为空的值
            for col in numeric_missing:
                if df_filled[col].isnull().any():
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                    df_filled[col] = df_filled[col].fillna(method='bfill')

        return df_filled

    @staticmethod
    def mean_imputation(df):
        """均值填充"""
        return ImputationMethods.fill_all_columns(df, strategy='mean')

    @staticmethod
    def median_imputation(df):
        """中位数填充"""
        return ImputationMethods.fill_all_columns(df, strategy='median')

    @staticmethod
    def mode_imputation(df):
        """众数填充"""
        return ImputationMethods.fill_all_columns(df, strategy='mode')

    @staticmethod
    def forward_fill(df):
        """前向填充"""
        return ImputationMethods.fill_all_columns(df, strategy='forward_fill')

    @staticmethod
    def backward_fill(df):
        """后向填充"""
        return ImputationMethods.fill_all_columns(df, strategy='backward_fill')

    @staticmethod
    def linear_interpolation(df):
        """线性插值"""
        return ImputationMethods.fill_all_columns(df, strategy='linear')

    @staticmethod
    def polynomial_interpolation(df):
        """多项式插值"""
        return ImputationMethods.fill_all_columns(df, strategy='polynomial')

    @staticmethod
    def spline_interpolation(df):
        """样条插值"""
        return ImputationMethods.fill_all_columns(df, strategy='spline')

    @staticmethod
    def knn_imputation(df, n_neighbors=5):
        """KNN填充 - 仅适用于数值列"""
        df_filled = df.copy()

        # 只对数值列进行KNN填充
        if numeric_missing:
            numeric_data = df_filled[numeric_missing]
            if not numeric_data.empty:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                filled_values = imputer.fit_transform(numeric_data)
                df_filled[numeric_missing] = filled_values

        # 其他类型列使用合适的方法填充
        if time_missing:
            df_filled[time_missing] = df_filled[time_missing].fillna(method='ffill')
            df_filled[time_missing] = df_filled[time_missing].fillna(method='bfill')

        if categorical_missing:
            for col in categorical_missing:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df_filled[col].fillna(mode_val[0], inplace=True)

        return df_filled


# 修改评估函数中的代码
def evaluate_imputation(original_data, filled_data, method_name):
    """评估填充效果"""
    # 计算所有列的缺失值总和 - 修复字符串和列表连接问题
    if time_missing:
        all_missing_cols = [time_missing] + categorical_missing + numeric_missing
    else:
        all_missing_cols = categorical_missing + numeric_missing

    results = {
        'method': method_name,
        'missing_before': original_data[all_missing_cols].isnull().sum().sum(),
        'missing_after': filled_data[all_missing_cols].isnull().sum().sum(),
        'mean_change': {},
        'std_change': {},
        'variance_ratio': {}
    }

    # 只评估数值列
    for col in numeric_missing:
        if original_data[col].notna().sum() > 0:
            orig_mean = original_data[col].mean()
            filled_mean = filled_data[col].mean()
            orig_std = original_data[col].std()
            filled_std = filled_data[col].std()

            # 避免除零错误
            if orig_mean != 0:
                results['mean_change'][col] = abs(orig_mean - filled_mean) / orig_mean * 100
            else:
                results['mean_change'][col] = abs(orig_mean - filled_mean) * 100

            if orig_std != 0:
                results['std_change'][col] = abs(orig_std - filled_std) / orig_std * 100
                results['variance_ratio'][col] = filled_std / orig_std
            else:
                results['std_change'][col] = abs(orig_std - filled_std) * 100
                results['variance_ratio'][col] = 1.0 if filled_std == 0 else float('inf')

    return results


# 执行所有填充方法
print("\n" + "=" * 50)
print("开始执行各种填充方法")
print("=" * 50)

methods = {
    'Mean': ImputationMethods.mean_imputation,
    'Median': ImputationMethods.median_imputation,
    'Mode': ImputationMethods.mode_imputation,
    'Forward Fill': ImputationMethods.forward_fill,
    'Backward Fill': ImputationMethods.backward_fill,
    'Linear Interpolation': ImputationMethods.linear_interpolation,
    'Polynomial Interpolation': ImputationMethods.polynomial_interpolation,
    'Spline Interpolation': ImputationMethods.spline_interpolation,
    'KNN Imputation': ImputationMethods.knn_imputation
}

results = {}
filled_datasets = {}

for method_name, method_func in methods.items():
    print(f"执行 {method_name}...")
    try:
        filled_data = method_func(data)
        results[method_name] = evaluate_imputation(data, filled_data, method_name)
        filled_datasets[method_name] = filled_data
        print(f"✓ {method_name} 完成")
    except Exception as e:
        print(f"✗ {method_name} 失败: {str(e)}")

# 以下部分保持不变
# 创建评估结果表格
print("\n" + "=" * 50)
print("填充方法效果对比")
print("=" * 50)

comparison_df = []
for method_name, result in results.items():
    row = {
        'Method': method_name,
        'Missing_Before': result['missing_before'],
        'Missing_After': result['missing_after'],
        'Completion_Rate(%)': (1 - result['missing_after'] / result['missing_before']) * 100 if result[
                                                                                                    'missing_before'] > 0 else 100
    }
    # 计算平均变化率
    if result['mean_change']:
        row['Avg_Mean_Change(%)'] = np.mean(list(result['mean_change'].values()))
        row['Avg_Std_Change(%)'] = np.mean(list(result['std_change'].values()))
        row['Avg_Variance_Ratio'] = np.mean(list(result['variance_ratio'].values()))
    comparison_df.append(row)

comparison_df = pd.DataFrame(comparison_df)
print(comparison_df)
missing_columns = numeric_missing

# 详细分析各列的填充效果
print("\n" + "=" * 50)
print("各列详细填充效果分析")
print("=" * 50)

for col in missing_columns:
    print(f"\n列: {col}")
    print("-" * 30)
    col_comparison = []

    for method_name, result in results.items():
        if col in result['mean_change']:
            col_comparison.append({
                'Method': method_name,
                'Mean_Change(%)': result['mean_change'][col],
                'Std_Change(%)': result['std_change'][col],
                'Variance_Ratio': result['variance_ratio'][col]
            })

    col_df = pd.DataFrame(col_comparison)
    if not col_df.empty:
        print(col_df.round(4))

# 可视化对比
plt.figure(figsize=(15, 10))
# 修改可视化部分
plt.figure(figsize=(15, 10))

# 检查是否有结果可以绘图
if comparison_df.empty:
    plt.text(0.5, 0.5, "没有成功完成的填充方法，无法生成图表",
             horizontalalignment='center', verticalalignment='center',
             fontsize=16, transform=plt.gca().transAxes)
else:
    # 子图1: 完成率对比
    plt.subplot(2, 2, 1)
    plt.bar(comparison_df['Method'], comparison_df['Completion_Rate(%)'])
    plt.title('Fill Completion Rate by Method')
    plt.xticks(rotation=45)
    plt.ylabel('Completion Rate (%)')

    # 子图2: 平均均值变化
    plt.subplot(2, 2, 2)
    if 'Avg_Mean_Change(%)' in comparison_df.columns:
        plt.bar(comparison_df['Method'], comparison_df['Avg_Mean_Change(%)'])
        plt.title('Average Mean Change by Method')
        plt.xticks(rotation=45)
        plt.ylabel('Mean Change (%)')

    # 子图3: 平均标准差变化
    plt.subplot(2, 2, 3)
    if 'Avg_Std_Change(%)' in comparison_df.columns:
        plt.bar(comparison_df['Method'], comparison_df['Avg_Std_Change(%)'])
        plt.title('Average Std Change by Method')
        plt.xticks(rotation=45)
        plt.ylabel('Std Change (%)')

    # 子图4: 方差比率
    plt.subplot(2, 2, 4)
    if 'Avg_Variance_Ratio' in comparison_df.columns:
        plt.bar(comparison_df['Method'], comparison_df['Avg_Variance_Ratio'])
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Original Variance')
        plt.title('Average Variance Ratio by Method')
        plt.xticks(rotation=45)
        plt.ylabel('Variance Ratio')
        plt.legend()

    plt.tight_layout()
    plt.savefig('imputation_methods_comparison.png')
    plt.show()

# 推荐最优方法
print("\n" + "=" * 50)
print("最优填充方法推荐")
print("=" * 50)

# 综合评分 (较低的均值变化和接近1的方差比率得分更高)
scoring_df = comparison_df.copy()
if 'Avg_Mean_Change(%)' in scoring_df.columns:
    scoring_df['Score'] = 0

    # 完成率权重 (30%)
    scoring_df['Score'] += (scoring_df['Completion_Rate(%)'] / 100) * 0.3

    # 均值变化越小越好 (40%)
    max_mean_change = scoring_df['Avg_Mean_Change(%)'].max()
    scoring_df['Score'] += (1 - scoring_df['Avg_Mean_Change(%)'] / max_mean_change) * 0.4

    # 方差比率接近1越好 (30%)
    scoring_df['Variance_Score'] = 1 - abs(scoring_df['Avg_Variance_Ratio'] - 1)
    scoring_df['Score'] += scoring_df['Variance_Score'] * 0.3

    # 排序
    scoring_df = scoring_df.sort_values('Score', ascending=False)

    print("综合评分排名:")
    print(scoring_df[['Method', 'Score', 'Completion_Rate(%)', 'Avg_Mean_Change(%)', 'Avg_Variance_Ratio']].round(4))

    best_method = scoring_df.iloc[0]['Method']
    print(f"\n推荐的最优填充方法: {best_method}")

    # 应用最优方法
    best_filled_data = filled_datasets[best_method]

    print(f"\n最优方法 ({best_method}) 填充后数据概览:")
    print(f"数据形状: {best_filled_data.shape}")
    print(f"剩余缺失值: {best_filled_data.isnull().sum().sum()}")

    # 保存填充后的数据
    output_filename = f'2025_9_20_filled_data_{best_method.lower().replace(" ", "_")}.xlsx'
    #output_filename = f'Subject1_2025_9_20_filled_data_{best_method.lower().replace(" ", "_")}.xlsx'
    best_filled_data.to_excel(output_filename, index=False)
    print(f"最优填充后的数据已保存为: {output_filename}")

print("\n" + "=" * 50)
print("分析完成!")
print("=" * 50)
