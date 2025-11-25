
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

# 机器学习模型
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


import warnings

warnings.filterwarnings('ignore')
######################################
#1修改目标变量
#2修改保存的文件名，按照序号依次对应，['疲劳分类标签', '血糖分类标签', '水合状态分类标签', '乳酸分类标签', '肌肉疲劳分类标签', '蛋白供应分类标签']['疲劳分类标签', '血糖分类标签', '水合状态分类标签', '乳酸分类标签', '肌肉疲劳分类标签', '蛋白供应分类标签']
####################################
num = 5  #自行修改
# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("机器学习完整Pipeline: EDA → 预处理 → 模型对比 → 最优选择")
print("=" * 80)

# 1. 数据加载和准备
print("\n1. 数据加载和准备")
print("-" * 50)

# 读取数据

data = pd.read_excel('2025_9_20_filled_data_linear_interpolation.xlsx')
print("✓ 使用填充后的数据")

print(f"数据形状: {data.shape}")
print(data.columns)
# 定义特征和目标变量
feature_cols = ['性别','Na (mM)', 'K (mM)', 'Glucose (uM)',
       'Lactate (mM)', 'SUN (mM)', 'NH4 (mM)', 'Cortisol', 'Tes', 'MDF', 'MEF']
feature_cols = [col for col in feature_cols if col in data.columns]

target_cols = [col for col in data.columns if '分类标签' in col]
print(f"特征变量: {feature_cols}")
print(f"目标变量: {target_cols}")

# 分别选择6个目标变量: ['疲劳分类标签', '血糖分类标签', '水合状态分类标签', '乳酸分类标签', '肌肉疲劳分类标签', '蛋白供应分类标签']进行建模
main_target = '肌肉疲劳分类标签'  # 可根据需要更改目标变量
if main_target not in data.columns:
    raise ValueError(f"目标变量 '{main_target}' 不在数据集中，请检查列名。")
print(f"当前选择的目标变量: {main_target}")


# ==================== 新增EDA探索性数据分析部分 ====================
print("\n" + "=" * 80)
print("EDA 探索性数据分析")
print("=" * 80)

# 2. 数据概览
print("\n2. 数据基本信息")
print("-" * 50)

print("数据集基本信息:")
print(f"数据形状: {data.shape}")
print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(data)}")

# 数据类型和缺失值信息
print("\n数据类型和缺失值:")
info_df = pd.DataFrame({
    '数据类型': data[feature_cols + [main_target]].dtypes,
    '缺失值数量': data[feature_cols + [main_target]].isnull().sum(),
    '缺失值比例(%)': (data[feature_cols + [main_target]].isnull().sum() / len(data) * 100).round(2),
    '唯一值数量': data[feature_cols + [main_target]].nunique()
})
print(info_df)

# 基本统计信息
print("\n特征变量描述性统计:")
desc_stats = data[feature_cols].describe()
print(desc_stats.round(4))

# 3. 目标变量分析
print("\n3. 目标变量分析")
print("-" * 50)

# 目标变量分布
target_counts = data[main_target].value_counts()
target_props = data[main_target].value_counts(normalize=True)

print("目标变量分布:")
target_summary = pd.DataFrame({
    '数量': target_counts,
    '比例(%)': (target_props * 100).round(2)
})
print(target_summary)

# 目标变量可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 柱状图
axes[0].bar(target_counts.index, target_counts.values)
axes[0].set_title(f'{main_target} 分布')
axes[0].set_xlabel('类别')
axes[0].set_ylabel('样本数量')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')

# 饼图
axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title(f'{main_target} 比例分布')

# 如果有多个目标变量，显示相关性
if len(target_cols) > 1:
    target_corr = data[target_cols].corr()
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, ax=axes[2])
    axes[2].set_title('目标变量间相关性')
else:
    axes[2].text(0.5, 0.5, '只有一个目标变量', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('目标变量相关性分析')

plt.tight_layout()
plt.savefig('target_variable_analysis.png', dpi=300)
plt.show()
#保存绘图的数据到csv
target_summary.to_csv('target_variable_summary.csv', encoding='utf-8-sig')


# 4. 特征变量分布分析
print("\n4. 特征变量分布分析")
print("-" * 50)

# 计算需要的子图数量
n_features = len(feature_cols)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

# 特征分布直方图
print("绘制特征分布直方图...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes

for i, col in enumerate(feature_cols):
    if i < len(axes):
        # 直方图和KDE
        axes[i].hist(data[col].dropna(), bins=30, density=True, alpha=0.7, edgecolor='black')

        # 添加KDE曲线
        try:
            data[col].dropna().plot.density(ax=axes[i], color='red', linewidth=2)
        except:
            pass

        axes[i].set_title(f'{col} 分布')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('密度')
        axes[i].grid(True, alpha=0.3)

        # 添加统计信息
        mean_val = data[col].mean()
        median_val = data[col].median()
        axes[i].axvline(mean_val, color='green', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.2f}')
        axes[i].axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'中位数: {median_val:.2f}')
        axes[i].legend(fontsize=8)

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('feature_distribution_analysis.png', dpi=300)
plt.show()
#保存绘图的数据到csv
desc_stats.to_csv('feature_descriptive_statistics.csv', encoding='utf-8-sig')

# 5. 箱线图分析
print("\n5. 箱线图分析（异常值检测）")
print("-" * 50)

print("绘制箱线图分析异常值...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes

outlier_summary = {}

for i, col in enumerate(feature_cols):
    if i < len(axes):
        # 箱线图
        box_plot = axes[i].boxplot(data[col].dropna(), patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')

        axes[i].set_title(f'{col} 箱线图')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

        # 计算异常值
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(data)) * 100

        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        # 添加异常值信息
        axes[i].text(0.02, 0.98, f'异常值: {outlier_count} ({outlier_percent:.1f}%)',
                     transform=axes[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 隐藏多余的子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('boxplot_outlier_analysis.png', dpi=300)
plt.show()

# 异常值统计
print("\n异常值统计总结:")
outlier_df = pd.DataFrame(outlier_summary).T
outlier_df.columns = ['异常值数量', '异常值比例(%)', '下界', '上界']
outlier_df['异常值比例(%)'] = outlier_df['异常值比例(%)'].round(2)
print(outlier_df)

# 6. 特征相关性分析
print("\n6. 特征相关性分析")
print("-" * 50)

print("计算特征间相关性...")
correlation_matrix = data[feature_cols].corr()

# 相关性热力图
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.3f')
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.savefig('feature_correlation_matrix.png', dpi=300)
plt.show()
#保存绘图的数据到csv
correlation_matrix.to_csv('feature_correlation_matrix.csv', encoding='utf-8-sig')


# 高相关性特征对，
print("\n高相关性特征对 (|r| > 0.9):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
    print(high_corr_df)
else:
    print("没有发现高相关性特征对")

#对高相关特征进行处理
if high_corr_pairs:
    to_remove = set()
    for pair in high_corr_pairs:
        # 简单策略：移除相关性较高对中的第二个特征
        to_remove.add(pair['feature2'])
    print(f"\n建议移除以下高相关性特征以减少多重共线性: {to_remove}")
    feature_cols = [col for col in feature_cols if col not in to_remove]
    print(f"更新后的特征列表: {feature_cols}")
else:
    print("无需移除任何特征")

# 7. 特征与目标变量关系分析
print("\n7. 特征与目标变量关系分析")
print("-" * 50)

# 不同类别下的特征分布对比
unique_targets = data[main_target].unique()
n_targets = len(unique_targets)

print(f"绘制不同{main_target}类别下的特征分布对比...")

# 为每个特征创建分类对比图
for idx, col in enumerate(feature_cols[:6]):  # 只显示前6个特征避免图太多
    plt.figure(figsize=(15, 5))

    # 小提琴图
    plt.subplot(1, 3, 1)
    sns.violinplot(data=data, x=main_target, y=col)
    plt.title(f'{col} - 小提琴图')
    plt.xticks(rotation=45)

    # 箱线图
    plt.subplot(1, 3, 2)
    sns.boxplot(data=data, x=main_target, y=col)
    plt.title(f'{col} - 箱线图对比')
    plt.xticks(rotation=45)

    # 直方图叠加
    plt.subplot(1, 3, 3)
    for target in unique_targets:
        subset = data[data[main_target] == target][col].dropna()
        plt.hist(subset, alpha=0.6, label=f'{target} (n={len(subset)})', bins=20)
    plt.xlabel(col)
    plt.ylabel('频次')
    plt.title(f'{col} - 分布对比')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'feature_{col}_by_{main_target}.png', dpi=300)
    plt.show()
#保存绘图的数据到csv
# 计算各类别下的均值和标准差
feature_target_summary = {}
for col in feature_cols:
    stats = []
    for target in unique_targets:
        subset = data[data[main_target] == target][col]
        stats.append({
            '类别': target,
            '均值': subset.mean(),
            '标准差': subset.std(),
            '样本数': len(subset)
        })
    feature_target_summary[col] = pd.DataFrame(stats)
    feature_target_summary[col].to_csv(f'feature_{col}_by_{main_target}_summary.csv', encoding='utf-8-sig', index=False)
    print(f"已保存 {col} 按 {main_target} 分类的统计信息到 CSV 文件")

# 8. 特征与目标变量相关性
print("\n8. 特征与目标变量相关性")
print("-" * 50)

# 如果目标变量是数值型，计算相关性
if data[main_target].dtype in ['int64', 'float64']:
    target_correlation = data[feature_cols + [main_target]].corr()[main_target].drop(main_target).sort_values(key=abs,
                                                                                                              ascending=False)

    print("特征与目标变量相关性:")
    print(target_correlation)

    # 可视化特征与目标变量相关性
    plt.figure(figsize=(10, 8))
    target_correlation.plot(kind='barh')
    plt.title(f'特征与{main_target}的相关性')
    plt.xlabel('相关系数')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'feature_target_correlation_{main_target}.png', dpi=300)
    plt.show()
    # 保存绘图的数据到csv
    target_correlation.to_csv(f'feature_target_correlation_{main_target}.csv', encoding='utf-8-sig')


else:
    # 如果是分类变量，使用方差分析
    from scipy import stats

    print("特征与目标变量关联性分析 (F-统计量):")
    f_stats = []
    p_values = []

    for col in feature_cols:
        groups = [data[data[main_target] == target][col].dropna() for target in unique_targets]
        f_stat, p_val = stats.f_oneway(*groups)
        f_stats.append(f_stat)
        p_values.append(p_val)

    anova_results = pd.DataFrame({
        'Feature': feature_cols,
        'F_statistic': f_stats,
        'p_value': p_values,
        'significant': ['是' if p < 0.05 else '否' for p in p_values]
    }).sort_values('F_statistic', ascending=False)

    print(anova_results)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

print("=== 增强版特征关系分析 ===")
# 确保数据准备
if 'data' not in locals():
    print("请先准备数据变量")
else:
    # 数据基本信息
    print(f"数据集形状: {data.shape}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"目标变量: {main_target}")
    print(f"类别分布:\n{data[main_target].value_counts()}")

# 创建类别名称映射
if 'class_names' in locals():
    class_mapping = {i: class_names[i] for i in range(len(class_names))}
    data_plot = data.copy()
    data_plot[main_target + '_name'] = data_plot[main_target].map(class_mapping)
    hue_col = main_target + '_name'
else:
    hue_col = main_target
    data_plot = data.copy()


# 2. 增强版 Pairplot
def enhanced_pairplot(data, feature_cols, hue_col, main_target):
    """创建增强版的特征关系图"""

    # 计算需要的子图数量
    n_features = len(feature_cols)

    # 创建自定义的pairplot
    fig, axes = plt.subplots(n_features, n_features, figsize=(4 * n_features, 4 * n_features))

    # 获取类别信息
    unique_classes = data[main_target].unique()
    colors = sns.color_palette("husl", len(unique_classes))
    class_colors = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                # 对角线：分布图
                for cls in unique_classes:
                    class_data = data[data[main_target] == cls][feature_cols[i]]

                    # 绘制直方图和核密度估计
                    ax.hist(class_data, bins=20, alpha=0.6,
                            label=f'{hue_col}: {cls}' if hue_col == main_target else f'{class_names[cls] if "class_names" in locals() else cls}',
                            color=class_colors[cls], density=True)

                    # 添加核密度估计曲线
                    if len(class_data) > 1:
                        kde_x = np.linspace(class_data.min(), class_data.max(), 100)
                        kde = stats.gaussian_kde(class_data)
                        ax.plot(kde_x, kde(kde_x), color=class_colors[cls], linewidth=2, alpha=0.8)

                ax.set_xlabel(feature_cols[i])
                ax.set_ylabel('密度')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # 添加统计信息
                overall_mean = data[feature_cols[i]].mean()
                overall_std = data[feature_cols[i]].std()
                ax.axvline(overall_mean, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(0.02, 0.98, f'μ={overall_mean:.2f}\nσ={overall_std:.2f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            else:
                # 非对角线：散点图
                for cls in unique_classes:
                    class_data = data[data[main_target] == cls]
                    x_data = class_data[feature_cols[j]]
                    y_data = class_data[feature_cols[i]]

                    # 散点图
                    ax.scatter(x_data, y_data, alpha=0.6, s=20,
                               color=class_colors[cls],
                               label=f'{class_names[cls] if "class_names" in locals() else cls}')

                    # 添加回归线
                    if len(x_data) > 1 and len(y_data) > 1:
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                            line_x = np.array([x_data.min(), x_data.max()])
                            line_y = slope * line_x + intercept
                            ax.plot(line_x, line_y, color=class_colors[cls],
                                    linestyle='-', alpha=0.8, linewidth=1.5)
                        except:
                            pass

                ax.set_xlabel(feature_cols[j])
                ax.set_ylabel(feature_cols[i])
                ax.grid(True, alpha=0.3)

                # 计算总体相关性
                corr_coef = data[feature_cols[j]].corr(data[feature_cols[i]])
                ax.text(0.02, 0.98, f'r={corr_coef:.3f}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

                # 只在第一行显示图例
                if i == 0 and j == 1:
                    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


# 3. 绘制增强版pairplot
print("绘制增强版特征关系图...")
enhanced_fig = enhanced_pairplot(data_plot, feature_cols, hue_col, main_target)
enhanced_fig.suptitle('增强版特征两两关系分析\n(包含分布、相关性和回归线)',
                      fontsize=16, fontweight='bold', y=0.995)
enhanced_fig.savefig('enhanced_pairplot_features.png', dpi=300, bbox_inches='tight')
plt.show()
#保存绘图的数据到csv
# 计算并保存相关性矩阵
corr_matrix = data[feature_cols].corr()
corr_matrix.to_csv('enhanced_pairplot_correlation_matrix.csv', encoding='utf-8-sig')
# 计算并保存各特征的描述性统计
desc_stats = data[feature_cols].describe().round(4)
desc_stats.to_csv('enhanced_pairplot_descriptive_statistics.csv', encoding='utf-8-sig')



# 4. 特征相关性热力图分析
print("绘制特征相关性分析...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 4.1 总体相关性矩阵
corr_matrix = data[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

im1 = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                  square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax1)
ax1.set_title('特征相关性矩阵 (总体)', fontsize=14, fontweight='bold')

# 4.2 按类别的相关性差异
unique_classes = data[main_target].unique()
if len(unique_classes) >= 2:
    class_corr_diff = {}
    for i, cls in enumerate(unique_classes[:2]):  # 比较前两个类别
        class_data = data[data[main_target] == cls][feature_cols]
        class_corr_diff[cls] = class_data.corr()

    # 计算相关性差异
    corr_diff = class_corr_diff[unique_classes[0]] - class_corr_diff[unique_classes[1]]

    im2 = sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0,
                      square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax2)
    class_0_name = class_names[unique_classes[0]] if 'class_names' in locals() else unique_classes[0]
    class_1_name = class_names[unique_classes[1]] if 'class_names' in locals() else unique_classes[1]
    ax2.set_title(f'类别间相关性差异\n({class_0_name} - {class_1_name})', fontsize=14, fontweight='bold')

# 4.3 特征与目标变量的关系强度
target_corr = {}
for feature in feature_cols:
    correlations = []
    for cls in unique_classes:
        class_data = data[data[main_target] == cls][feature]
        # 计算与类别编码的相关性
        class_encoded = (data[main_target] == cls).astype(int)
        corr = data[feature].corr(class_encoded)
        correlations.append(abs(corr))  # 使用绝对值
    target_corr[feature] = max(correlations)  # 取最大相关性

target_corr_df = pd.DataFrame(list(target_corr.items()), columns=['Feature', 'Target_Correlation'])
target_corr_df = target_corr_df.sort_values('Target_Correlation', ascending=True)

bars = ax3.barh(target_corr_df['Feature'], target_corr_df['Target_Correlation'],
                color='steelblue', alpha=0.7)
ax3.set_xlabel('与目标变量的最大相关性')
ax3.set_title('特征与目标变量关系强度', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 添加数值标签
for bar, corr in zip(bars, target_corr_df['Target_Correlation']):
    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{corr:.3f}', ha='left', va='center', fontsize=10)

# 4.4 特征分离度分析（各类别间特征值的分离程度）
separation_scores = {}
for feature in feature_cols:
    class_means = []
    class_stds = []

    for cls in unique_classes:
        class_data = data[data[main_target] == cls][feature]
        class_means.append(class_data.mean())
        class_stds.append(class_data.std())

    # 计算分离度：类间方差 / 类内方差的平均
    between_var = np.var(class_means)
    within_var = np.mean([std ** 2 for std in class_stds])
    separation = between_var / (within_var + 1e-6)  # 避免除零
    separation_scores[feature] = separation

sep_df = pd.DataFrame(list(separation_scores.items()), columns=['Feature', 'Separation_Score'])
sep_df = sep_df.sort_values('Separation_Score', ascending=True)

bars2 = ax4.barh(sep_df['Feature'], sep_df['Separation_Score'],
                 color='coral', alpha=0.7)
ax4.set_xlabel('分离度得分 (类间方差/类内方差)')
ax4.set_title('特征类别分离度分析', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars2, sep_df['Separation_Score']):
    ax4.text(bar.get_width() + max(sep_df['Separation_Score']) * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f'{score:.2f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
#保存绘图的数据到csv
corr_matrix.to_csv('feature_correlation_matrix_overall.csv', encoding='utf-8-sig')
target_corr_df.to_csv('feature_target_correlation_strength.csv', encoding='utf-8-sig', index=False)

# 5. 分类别的特征分布对比
print("绘制分类别特征分布对比...")
n_cols = 3
n_rows = (len(feature_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.ravel() if len(feature_cols) > 1 else [axes]

for idx, feature in enumerate(feature_cols):
    ax = axes[idx]

    # 为每个类别绘制箱线图和小提琴图的组合
    class_data = []
    class_labels = []

    for cls in unique_classes:
        class_feature_data = data[data[main_target] == cls][feature]
        class_data.append(class_feature_data)
        class_labels.append(class_names[cls] if 'class_names' in locals() else f'Class {cls}')

    # 绘制小提琴图
    parts = ax.violinplot(class_data, positions=range(len(unique_classes)),
                          showmeans=True, showmedians=True)

    # 自定义颜色
    colors = sns.color_palette("husl", len(unique_classes))
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # 设置样式
    ax.set_xticks(range(len(unique_classes)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} 分布对比')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = ""
    for i, (cls, class_feature_data) in enumerate(zip(unique_classes, class_data)):
        mean_val = np.mean(class_feature_data)
        std_val = np.std(class_feature_data)
        median_val = np.median(class_feature_data)
        stats_text += f"{class_labels[i]}: μ={mean_val:.2f}, σ={std_val:.2f}, Med={median_val:.2f}\n"

    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 进行统计检验（ANOVA）
    if len(class_data) > 1 and all(len(cd) > 1 for cd in class_data):
        try:
            f_stat, p_value = stats.f_oneway(*class_data)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            ax.text(0.98, 0.98, f'ANOVA: p={p_value:.3f} {significance}',
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        except:
            pass

# 隐藏多余的子图
for idx in range(len(feature_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('feature_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
#保存绘图的数据到csv
for feature in feature_cols:
    stats = []
    for cls in unique_classes:
        class_feature_data = data[data[main_target] == cls][feature]
        stats.append({
            '类别': class_names[cls] if 'class_names' in locals() else f'Class {cls}',
            '均值': class_feature_data.mean(),
            '标准差': class_feature_data.std(),
            '中位数': class_feature_data.median(),
            '样本数': len(class_feature_data)
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'feature_{feature}_distribution_comparison.csv', encoding='utf-8-sig', index=False)
    print(f"已保存 {feature} 按 {main_target} 分类的统计信息到 CSV 文件")
# 计算并保存ANOVA结果
anova_results = []
for feature in feature_cols:
    class_data = [data[data[main_target] == cls][feature] for cls in unique_classes]
    if len(class_data) > 1 and all(len(cd) > 1 for cd in class_data):
        try:
            f_stat, p_value = stats.f_oneway(*class_data)
            anova_results.append({
                'Feature': feature,
                'F_statistic': f_stat,
                'p_value': p_value,
                'significant': '是' if p_value < 0.05 else '否'
            })
        except:
            pass
anova_df = pd.DataFrame(anova_results)
anova_df.to_csv('feature_anova_results.csv', encoding='utf-8-sig', index=False)
print("已保存 ANOVA 结果到 CSV 文件")


# 6. 特征重要性和可分离性综合分析
print("生成特征分析综合报告...")

# 计算各种特征评分
feature_analysis = pd.DataFrame({
    'Feature': feature_cols,
    'Target_Correlation': [target_corr[f] for f in feature_cols],
    'Separation_Score': [separation_scores[f] for f in feature_cols],
    'Variance': [data[f].var() for f in feature_cols],
    'CV': [data[f].std() / abs(data[f].mean()) if abs(data[f].mean()) > 1e-6 else 0 for f in feature_cols]
})

# 标准化评分
scaler = StandardScaler()
feature_analysis['Target_Correlation_Norm'] = scaler.fit_transform(feature_analysis[['Target_Correlation']])
feature_analysis['Separation_Score_Norm'] = scaler.fit_transform(feature_analysis[['Separation_Score']])
feature_analysis['Variance_Norm'] = scaler.fit_transform(feature_analysis[['Variance']])

# 计算综合评分
feature_analysis['Composite_Score'] = (
        0.4 * feature_analysis['Target_Correlation_Norm'] +
        0.4 * feature_analysis['Separation_Score_Norm'] +
        0.2 * feature_analysis['Variance_Norm']
)

feature_analysis = feature_analysis.sort_values('Composite_Score', ascending=False)

# 可视化综合分析
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 综合评分排序
bars1 = ax1.barh(feature_analysis['Feature'], feature_analysis['Composite_Score'],
                 color='steelblue', alpha=0.7)
ax1.set_xlabel('综合评分 (标准化)')
ax1.set_title('特征综合重要性排序', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 散点图：相关性 vs 分离度
scatter = ax2.scatter(feature_analysis['Target_Correlation'],
                      feature_analysis['Separation_Score'],
                      s=feature_analysis['Variance'] * 10,  # 用方差控制点大小
                      alpha=0.7, c=range(len(feature_analysis)), cmap='viridis')

for i, feature in enumerate(feature_analysis['Feature']):
    ax2.annotate(feature,
                 (feature_analysis.iloc[i]['Target_Correlation'],
                  feature_analysis.iloc[i]['Separation_Score']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('与目标变量相关性')
ax2.set_ylabel('类别分离度')
ax2.set_title('特征性能二维分析\n(气泡大小=方差)')
ax2.grid(True, alpha=0.3)

# 变异系数分析
bars3 = ax3.bar(feature_analysis['Feature'], feature_analysis['CV'],
                color='coral', alpha=0.7)
ax3.set_ylabel('变异系数 (CV)')
ax3.set_title('特征变异性分析', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# 雷达图 - 前5个重要特征
top_5_features = feature_analysis.head(5)
categories = ['目标相关性', '分离度', '方差', '变异系数']

# 数据标准化到0-1范围
radar_data = []
for _, row in top_5_features.iterrows():
    radar_values = [
        row['Target_Correlation'],
        row['Separation_Score'] / feature_analysis['Separation_Score'].max(),
        row['Variance'] / feature_analysis['Variance'].max(),
        min(row['CV'], 1.0)  # CV限制在1以内
    ]
    radar_data.append(radar_values)

# 简化的雷达图替代方案：堆叠条形图
bottom = np.zeros(len(top_5_features))
colors_radar = ['red', 'blue', 'green', 'orange']

for i, category in enumerate(categories):
    values = [rd[i] for rd in radar_data]
    ax4.bar(top_5_features['Feature'], values, bottom=bottom,
            label=category, color=colors_radar[i], alpha=0.7)
    bottom += values

ax4.set_ylabel('标准化得分 (累积)')
ax4.set_title('Top 5 特征多维度分析', fontsize=14, fontweight='bold')
ax4.legend(loc='upper left')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
#保存绘图的数据到csv
feature_analysis.to_csv('comprehensive_feature_analysis.csv', encoding='utf-8-sig', index=False)


# 7. 生成详细的文字报告
print("\n" + "=" * 80)
print("特征关系分析综合报告")
print("=" * 80)

print(f"\n📊 数据集概况:")
print(f"   - 样本数量: {data.shape[0]}")
print(f"   - 特征数量: {len(feature_cols)}")
print(f"   - 目标变量类别数: {len(unique_classes)}")

print(f"\n🏆 特征重要性排序 (Top 5):")
for i, (_, row) in enumerate(feature_analysis.head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}")
    print(f"      - 目标相关性: {row['Target_Correlation']:.3f}")
    print(f"      - 分离度得分: {row['Separation_Score']:.3f}")
    print(f"      - 方差: {row['Variance']:.3f}")
    print(f"      - 综合评分: {row['Composite_Score']:.3f}")

print(f"\n📈 类别分离性分析:")
for cls in unique_classes:
    cls_name = class_names[cls] if 'class_names' in locals() else f'类别 {cls}'
    cls_count = sum(data[main_target] == cls)
    cls_percent = cls_count / len(data) * 100
    print(f"   - {cls_name}: {cls_count} 样本 ({cls_percent:.1f}%)")

best_separating_feature = feature_analysis.loc[feature_analysis['Separation_Score'].idxmax(), 'Feature']
print(f"   - 最佳分离特征: {best_separating_feature}")
print(f"   - 分离度得分: {feature_analysis['Separation_Score'].max():.3f}")

print(f"\n⚠️  数据质量提醒:")
low_var_features = feature_analysis[feature_analysis['Variance'] < feature_analysis['Variance'].quantile(0.25)][
    'Feature'].tolist()
if low_var_features:
    print(f"   - 低方差特征 (可能信息量不足): {low_var_features}")

high_cv_features = feature_analysis[feature_analysis['CV'] > 1.0]['Feature'].tolist()
if high_cv_features:
    print(f"   - 高变异特征 (可能需要标准化): {high_cv_features}")

print("\n" + "=" * 80)
print("分析完成！所有图表已保存为 PNG 文件。")
print("=" * 80)

# 9. 数据质量报告
print("\n9. 数据质量报告")
print("-" * 50)

quality_report = {}

# 缺失值分析
missing_analysis = data[feature_cols].isnull().sum()
quality_report['missing_values'] = {
    'total_missing': missing_analysis.sum(),
    'features_with_missing': (missing_analysis > 0).sum(),
    'max_missing_percent': (missing_analysis.max() / len(data) * 100)
}

# 异常值分析
total_outliers = sum([info['count'] for info in outlier_summary.values()])
quality_report['outliers'] = {
    'total_outliers': total_outliers,
    'outlier_percent': (total_outliers / (len(data) * len(feature_cols)) * 100),
    'features_with_outliers': sum([1 for info in outlier_summary.values() if info['count'] > 0])
}

# 数据不平衡分析
target_imbalance = target_props.max() / target_props.min()
quality_report['class_imbalance'] = {
    'imbalance_ratio': target_imbalance,
    'is_imbalanced': target_imbalance > 2
}

# 特征多重共线性
high_corr_count = len(high_corr_pairs)
quality_report['multicollinearity'] = {
    'high_corr_pairs': high_corr_count,
    'potential_multicollinearity': high_corr_count > 0
}



print("数据质量总结:")
print(
    f"✓ 数据完整性: {(1 - quality_report['missing_values']['total_missing'] / (len(data) * len(feature_cols))) * 100:.1f}%")
print(f"✓ 异常值比例: {quality_report['outliers']['outlier_percent']:.2f}%")
print(f"✓ 高相关性特征对: {quality_report['multicollinearity']['high_corr_pairs']} 对")

# 数据质量可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 缺失值可视化
axes[0, 0].bar(range(len(missing_analysis)), missing_analysis.values)
axes[0, 0].set_title('各特征缺失值数量')
axes[0, 0].set_xticks(range(len(missing_analysis)))
axes[0, 0].set_xticklabels(missing_analysis.index, rotation=45)
axes[0, 0].set_ylabel('缺失值数量')

# 异常值可视化
outlier_counts = [info['count'] for info in outlier_summary.values()]
axes[0, 1].bar(range(len(outlier_counts)), outlier_counts)
axes[0, 1].set_title('各特征异常值数量')
axes[0, 1].set_xticks(range(len(feature_cols)))
axes[0, 1].set_xticklabels(feature_cols, rotation=45)
axes[0, 1].set_ylabel('异常值数量')

# 目标变量不平衡
axes[1, 0].bar(target_counts.index, target_counts.values)
axes[1, 0].set_title('目标变量类别分布')
axes[1, 0].set_ylabel('样本数量')

# 特征标准差
feature_std = data[feature_cols].std().sort_values(ascending=False)
axes[1, 1].bar(range(len(feature_std)), feature_std.values)
axes[1, 1].set_title('特征标准差')
axes[1, 1].set_xticks(range(len(feature_std)))
axes[1, 1].set_xticklabels(feature_std.index, rotation=45)
axes[1, 1].set_ylabel('标准差')

plt.tight_layout()
plt.savefig('data_quality_report.png', dpi=300, bbox_inches='tight')
plt.show()
#保存绘图的数据到csv
missing_analysis.to_csv('data_quality_missing_values.csv', encoding='utf-8-sig')
outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index')
outlier_df.to_csv('data_quality_outlier_summary.csv', encoding='utf-8-sig')
target_counts.to_csv('data_quality_target_distribution.csv', encoding='utf-8-sig')
feature_std.to_csv('data_quality_feature_std.csv', encoding='utf-8-sig')

print(f"\n🔍 EDA分析完成！发现了以下关键信息:")
print(f"   - 数据集包含 {len(feature_cols)} 个特征, {len(data)} 个样本")
print(f"   - 缺失值: {quality_report['missing_values']['total_missing']} 个")
print(f"   - 异常值: {total_outliers} 个 ({quality_report['outliers']['outlier_percent']:.2f}%)")
print(f"   - 多重共线性: {'存在' if quality_report['multicollinearity']['potential_multicollinearity'] else '无'}")

# ==================== 预处理和建模部分 ====================

# 2. 数据预处理
print("\n" + "=" * 80)
print("数据预处理")
print("=" * 80)

# 提取特征和目标
X = data[feature_cols].copy()
y = data[main_target].copy()

# 处理缺失值
print("处理缺失值...")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"处理前缺失值: {X.isnull().sum().sum()}")
print(f"处理后缺失值: {X_filled.isnull().sum().sum()}")

# 3. 异常值处理
print("\n3. 异常值检测和处理")
print("-" * 50)


def detect_outliers_iqr(df, column):
    """使用IQR方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound


# 检测异常值
outlier_info = {}
total_outliers = pd.Series([False] * len(X_filled))

for col in feature_cols:
    outliers, lower, upper = detect_outliers_iqr(X_filled, col)
    outlier_count = outliers.sum()
    outlier_percent = (outlier_count / len(X_filled)) * 100

    outlier_info[col] = {
        'count': outlier_count,
        'percentage': outlier_percent,
        'lower_bound': lower,
        'upper_bound': upper
    }

    total_outliers = total_outliers | outliers

print("异常值统计:")
for col, info in outlier_info.items():
    print(f"{col}: {info['count']} ({info['percentage']:.2f}%)")

print(f"\n总异常值样本数: {total_outliers.sum()} ({(total_outliers.sum() / len(X_filled)) * 100:.2f}%)")

# 异常值处理策略
outlier_threshold = 0.05  # 5%阈值
if (total_outliers.sum() / len(X_filled)) > outlier_threshold:
    print("\n异常值比例较高，使用Winsorizing方法处理...")
    # Winsorizing: 将异常值替换为分位数值
    X_clean = X_filled.copy()
    for col in feature_cols:
        outliers, lower, upper = detect_outliers_iqr(X_filled, col)
        X_clean.loc[X_clean[col] < lower, col] = lower
        X_clean.loc[X_clean[col] > upper, col] = upper
else:
    print("\n异常值比例较低，直接移除异常值...")
    # 移除异常值
    X_clean = X_filled[~total_outliers].copy()
    y_clean = y[~total_outliers].copy()

print(f"处理后数据形状: {X_clean.shape}")

# 4. 数据标准化
print("\n4. 数据标准化")
print("-" * 50)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)

print("标准化前后对比:")
comparison_df = pd.DataFrame({
    '原始均值': X_clean.mean(),
    '原始标准差': X_clean.std(),
    '标准化后均值': X_scaled.mean(),
    '标准化后标准差': X_scaled.std()
})
print(comparison_df.round(4))

# 5. PCA降维分析
print("\n5. PCA降维分析")
print("-" * 50)

# 执行PCA分析
pca_full = PCA()
pca_result = pca_full.fit_transform(X_scaled)

# 计算累积解释方差
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 可视化PCA结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 个体解释方差
ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
ax1.set_xlabel('主成分')
ax1.set_ylabel('解释方差比例')
ax1.set_title('各主成分解释方差比例')
ax1.grid(True, alpha=0.3)

# 累积解释方差
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
ax2.axhline(y=0.8, color='r', linestyle='--', label='80%')
ax2.axhline(y=0.9, color='g', linestyle='--', label='90%')
ax2.axhline(y=0.95, color='orange', linestyle='--', label='95%')
ax2.set_xlabel('主成分数量')
ax2.set_ylabel('累积解释方差比例')
ax2.set_title('累积解释方差比例')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300)
plt.show()
#保存绘图的数据到csv
pca_variance_df = pd.DataFrame({
    'Principal_Component': [f'PC{i + 1}' for i in range(len(explained_variance))],
    'Explained_Variance_Ratio': explained_variance,
    'Cumulative_Explained_Variance': cumulative_variance
})
pca_variance_df.to_csv('pca_variance_analysis.csv', encoding='utf-8-sig', index=False)


# PCA降维决策
n_features = len(feature_cols)
n_components_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1
n_components_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

print(f"原始特征数: {n_features}")
print(f"解释80%方差需要: {n_components_80} 个主成分")
print(f"解释90%方差需要: {n_components_90} 个主成分")
print(f"解释95%方差需要: {n_components_95} 个主成分")

# 降维决策
use_pca = False
if n_features > 10 and n_components_90 < n_features * 0.7:
    use_pca = True
    optimal_components = n_components_90
    print(f"\n✓ 建议使用PCA降维，保留{optimal_components}个主成分")

    pca = PCA(n_components=optimal_components)
    X_final = pd.DataFrame(pca.fit_transform(X_scaled),
                           columns=[f'PC{i + 1}' for i in range(optimal_components)])
else:
    print(f"\n✓ 不建议使用PCA降维，保持原始特征")
    X_final = X_scaled

print(f"最终特征维度: {X_final.shape}")

# 6. 数据集划分
print("\n6. 数据集划分")
print("-" * 50)

# 确保 X_final 和 y_final 定义一致
if 'X_clean' in locals() and 'y_clean' in locals():
    X_final = X_clean
    y_final = y_clean
else:
    X_final = X
    y_final = y

# 检查并确保样本数量一致
print(f"检查数据集: X_final形状: {X_final.shape}, y_final长度: {len(y_final)}")

if len(X_final) != len(y_final):
    print(f"⚠️ 发现样本数不一致！X_final: {len(X_final)}, y_final: {len(y_final)}")

    # 方法1: 取两者的交集 (推荐)
    common_indices = X_final.index.intersection(y_final.index) if hasattr(y_final, 'index') else None

    if common_indices is not None and len(common_indices) > 0:
        print(f"使用索引交集: {len(common_indices)} 个样本")
        X_final = X_final.loc[common_indices]
        y_final = y_final.loc[common_indices]
    else:
        # 方法2: 取前N个样本 (备选)
        min_samples = min(len(X_final), len(y_final))
        print(f"对齐到相同长度: {min_samples} 个样本")
        X_final = X_final.iloc[:min_samples] if hasattr(X_final, 'iloc') else X_final[:min_samples]
        y_final = y_final.iloc[:min_samples] if hasattr(y_final, 'iloc') else y_final[:min_samples]

# 重置索引确保一致性
X_final = X_final.reset_index(drop=True) if hasattr(X_final, 'reset_index') else X_final
if hasattr(y_final, 'reset_index'):
    y_final = y_final.reset_index(drop=True)

# 再次检查一致性
print(f"对齐后数据: X_final形状: {X_final.shape}, y_final长度: {len(y_final)}")
assert len(X_final) == len(y_final), "数据集样本数仍不一致!"

#保存处理之后的数据到csv
X_final.to_csv('final_features.csv', encoding='utf-8-sig', index=False)
y_final.to_csv('final_target.csv', encoding='utf-8-sig', index=False)
print("已分别保存最终特征和目标变量到 CSV 文件")
print(X_final.head())
print(y_final.head())
#最终特征和目标变量到一个 CSV 文件
final_data = pd.concat([X_final, y_final.reset_index(drop=True)], axis=1)
final_data.to_csv('final_dataset.csv', encoding='utf-8-sig', index=False)
print("已保存最终数据集到 CSV 文件")
# 使用分层抽样确保类别分布一致
# 然后再进行划分
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"训练集标签分布: {pd.Series(y_train).value_counts().to_dict()}")
print(f"测试集标签分布: {pd.Series(y_test).value_counts().to_dict()}")


# 7. 机器学习模型定义
print("\n7. 机器学习模型定义")
print("-" * 50)

# 定义13种机器学习模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100),
    'Support Vector Machine': SVC(probability=True),
    'Gaussian Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Multi-layer Perceptron': MLPClassifier(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'lightGBM': LGBMClassifier()
}

print(f"定义了{len(models)}种机器学习模型")

# 8. 模型训练和评估
print("\n8. 模型训练和评估")
print("-" * 50)

# 存储结果
results = {}
cv_scores = {}
predictions = {}

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("开始训练和评估模型...")
for name, model in models.items():
    print(f"训练 {name}...", end=' ')

    try:
        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') and len(
            np.unique(y_final)) == 2 else None

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 交叉验证
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        # 存储结果
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_score.mean(),
            'cv_std': cv_score.std()
        }

        cv_scores[name] = cv_score
        predictions[name] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}

        print("✓")

    except Exception as e:
        print(f"✗ 错误: {str(e)}")
        continue

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

# ------------- 2. 训练所有模型，保存分数 -------------------
print("\n=== 模型训练并缓存预测概率 ===")
y_test_bin = label_binarize(y_final, classes=sorted(y_final.unique()))
n_classes = y_test_bin.shape[1]

# 重新切分（保持与前面一致）
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
y_test_bin = label_binarize(y_test, classes=sorted(y_final.unique()))

model_scores = {}  # 存AUC
model_fpr_tpr = {}  # 存曲线 (fpr, tpr)
skip_models = []  # 无法画ROC的模型

for name, model in models.items():
    try:
        model.fit(X_train, y_train)

        # 取得“连续输出”以绘制 ROC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)  # shape = (n_samples, n_classes)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            # 若 decision_function 只给 (n_samples,)，需转成 (n_samples, n_classes)
            if y_score.ndim == 1:
                # 二分类才会遇到，但为了代码健壮性：
                y_score = np.column_stack([-y_score, y_score])
        else:
            print(f"⚠️  {name} 既无 predict_proba 也无 decision_function，跳过 ROC。")
            skip_models.append(name)
            continue

        # 计算 micro-average & macro-average AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_bin.ravel(), y_score.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        model_scores[name] = roc_auc
        model_fpr_tpr[name] = (fpr, tpr)
        print(f"✓ {name} - macro AUC: {roc_auc['macro']:.3f}")

    except Exception as e:
        print(f"✗ {name} 训练或预测出错: {e}")
        skip_models.append(name)
        continue


# ------------- 3. 绘制一张大图：macro-average ROC -----------------
print("\n=== 绘制 ROC 曲线 ===")
plt.figure(figsize=(10, 8))
colors = cycle(plt.cm.tab20.colors)  # 至少 20 种颜色

for (name, color) in zip(model_scores.keys(), colors):
    fpr, tpr = model_fpr_tpr[name]
    auc_val = model_scores[name]["macro"]
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        color=color,
        lw=2,
        label=f"{name} (AUC = {auc_val:.3f})"
    )

# 对角线
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Macro-Average ROC Curves (3-class, 12 Models)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("all_models_macro_roc.png", dpi=300)
plt.show()
#保存绘图的数据到csv
macro_roc_data = []
for name in model_scores.keys():
    fpr, tpr = model_fpr_tpr[name]
    for fp, tp in zip(fpr["macro"], tpr["macro"]):
        macro_roc_data.append({
            'Model': name,
            'FPR': fp,
            'TPR': tp,
            'AUC': model_scores[name]['macro']
        })
macro_roc_df = pd.DataFrame(macro_roc_data)
macro_roc_df.to_csv('all_models_macro_roc_data.csv', encoding='utf-8-sig', index=False)


# ------------- 4. （可选）再画 micro-average -----------------
plt.figure(figsize=(10, 8))
colors = cycle(plt.cm.Dark2.colors)

for (name, color) in zip(model_scores.keys(), colors):
    fpr, tpr = model_fpr_tpr[name]
    auc_val = model_scores[name]["micro"]
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        color=color,
        lw=2,
        label=f"{name} (AUC = {auc_val:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Micro-Average ROC Curves (3-class, 12 Models)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("all_models_micro_roc.png", dpi=300)
plt.show()
#保存绘图的数据到csv
micro_roc_data = []
for name in model_scores.keys():
    fpr, tpr = model_fpr_tpr[name]
    for fp, tp in zip(fpr["micro"], tpr["micro"]):
        micro_roc_data.append({
            'Model': name,
            'FPR': fp,
            'TPR': tp,
            'AUC': model_scores[name]['micro']
        })
micro_roc_df = pd.DataFrame(micro_roc_data)
micro_roc_df.to_csv('all_models_micro_roc_data.csv', encoding='utf-8-sig', index=False)


# ------------- 5. 简单汇总表 -----------------
print("\n=== 主要 AUC 汇总 (macro / micro) ===")
for name, scores in model_scores.items():
    print(f"{name:25s}  Macro AUC: {scores['macro']:.3f}  |  Micro AUC: {scores['micro']:.3f}")

if skip_models:
    print("\n⚠️  以下模型因缺少连续输出而未绘制 ROC：", ", ".join(skip_models))

# 9. 结果可视化对比
print("\n9. 模型性能可视化对比")
print("-" * 50)

# 创建结果DataFrame
results_df = pd.DataFrame(results).T
print("\n模型性能对比表:")
print(results_df.round(4))

# 性能对比可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
metric_names = ['准确率', '精确率', '召回率', 'F1分数', '交叉验证均值']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    if i < len(axes):
        data_to_plot = results_df[metric].sort_values(ascending=False)
        bars = axes[i].bar(range(len(data_to_plot)), data_to_plot.values, color='skyblue', alpha=0.8)
        axes[i].set_title(f'{name} 对比')
        axes[i].set_xticks(range(len(data_to_plot)))
        axes[i].set_xticklabels(data_to_plot.index, rotation=45, ha='right')
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)

        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 交叉验证分数箱线图
if len(cv_scores) > 0:
    axes[5].boxplot([cv_scores[name] for name in results.keys()],
                    labels=[name for name in results.keys()])
    axes[5].set_title('交叉验证分数分布')
    axes[5].set_xticklabels(results.keys(), rotation=45, ha='right')
    axes[5].set_ylabel('准确率')
    axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
plt.show()
#保存绘图的数据到csv
results_df.to_csv('model_performance_comparison.csv', encoding='utf-8-sig')
cv_scores_df = pd.DataFrame({name: scores for name, scores in cv_scores.items()})
cv_scores_df.to_csv('model_cross_validation_scores.csv', encoding='utf-8-sig', index=False)

# 10. 综合排名和最优模型选择
print("\n10. 综合排名和最优模型选择")
print("-" * 50)

# 计算综合得分
weights = {
    'accuracy': 0.3,
    'precision': 0.2,
    'recall': 0.2,
    'f1_score': 0.2,
    'cv_mean': 0.1
}

results_df['综合得分'] = 0
for metric, weight in weights.items():
    # 标准化到0-1范围
    normalized = (results_df[metric] - results_df[metric].min()) / (results_df[metric].max() - results_df[metric].min())
    results_df['综合得分'] += normalized * weight

# 排序
final_ranking = results_df.sort_values('综合得分', ascending=False)

print("最终模型排名:")
print("=" * 70)
ranking_display = final_ranking[['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean', '综合得分']].round(4)
for i, (name, row) in enumerate(ranking_display.iterrows(), 1):
    print(f"{i:2d}. {name:15s} | 综合得分: {row['综合得分']:.4f} | "
          f"准确率: {row['accuracy']:.4f} | F1: {row['f1_score']:.4f} | "
          f"CV: {row['cv_mean']:.4f}±{final_ranking.loc[name, 'cv_std']:.4f}")

# 选择最优模型
best_model_name = final_ranking.index[0]
best_model = models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\n🏆 最优模型: {best_model_name}")
print(f"   综合得分: {final_ranking.iloc[0]['综合得分']:.4f}")
print(f"   准确率: {final_ranking.iloc[0]['accuracy']:.4f}")
print(f"   F1分数: {final_ranking.iloc[0]['f1_score']:.4f}")

# 11. 模型详细分析
print("\n11. 模型详细分析")
print("-" * 50)
#12种模型中的混淆矩阵
for name in results.keys():
    y_pred = predictions[name]['y_pred']
    y_pred_proba = predictions[name]['y_pred_proba']

    print(f"\n🔍 {name} 模型详细分析")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'{name}_confusion_matrix.png', dpi=300)
    plt.show()
#保存绘图的数据到csv
    cm_df = pd.DataFrame(cm, index=np.unique(y_final), columns=np.unique(y_final))
    cm_df.to_csv(f'{name}_confusion_matrix.csv', encoding='utf-8-sig')
    # 分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    print(f"\n分类报告:\n{report_df.round(4)}")
    report_df.to_csv(f'{name}_classification_report.csv', encoding='utf-8-sig')

print("\n" + "=" * 80)
print("🎉 完整的机器学习Pipeline已完成!")
print("🔍 包含EDA分析 → 数据预处理 → 模型对比 → 结果分析")
print("📊 生成了详细的可视化图表和分析报告")
print("=" * 80)
#测试集预测结果，计算各个类别概率
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
test_results_df = pd.DataFrame({
    '真实标签': y_test,
    '预测标签': y_test_pred
})
if y_test_pred_proba is not None:
    for i in range(y_test_pred_proba.shape[1]):
        test_results_df[f'类别_{i}_概率'] = y_test_pred_proba[:, i]


test_results_df.to_csv('best_model_test_predictions.csv', encoding='utf-8-sig', index=False)
print(test_results_df.head())
print("测试集预测结果已保存到 'best_model_test_predictions.csv'")



# 12. 模型保存和新数据预测
print("\n12. 模型保存和新数据预测")
print("-" * 50)

# =================================================================
# 保存权重参数和预处理器
# =================================================================
import joblib
import os

try:
    # 确保保存目录存在
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # 使用f-string简化文件名生成
    joblib.dump(imputer, f"{save_dir}/{num}_imputer.pkl")
    joblib.dump(scaler, f"{save_dir}/{num}_scaler.pkl")
    print("✓ 预处理器 (imputer, scaler) 已保存。")

    # 保存PCA配置
    pca_config = {
        'use_pca': use_pca,
        'feature_cols': feature_cols
    }

    if use_pca:
        joblib.dump(pca, f"{save_dir}/{num}_pca.pkl")
        pca_config['optimal_components'] = optimal_components
        print("✓ PCA 对象已保存。")

    joblib.dump(pca_config, f"{save_dir}/{num}_pca_config.pkl")
    print("✓ PCA 配置已保存。")
    # 保留最优模型的权重参数
    joblib.dump(best_model, f"{save_dir}/{num}_best_model_{best_model_name.replace(' ', '_')}.pkl")
    print(f"✓ 最优模型 ({best_model_name}) 已保存。")

except Exception as e:
    print(f"保存模型时出错: {e}")








