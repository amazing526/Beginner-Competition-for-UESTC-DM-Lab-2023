import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

data_train = pd.read_csv("../data/recipes_train.csv")

i = 1
X = np.arange(0, 383)
# 设置画布
fig = plt.figure(figsize=(9, 9), dpi=80)
for name, entries in data_train.groupby("cuisine"):  # 按照cuisine列进行分组
    entries = entries.drop(columns=['id', "cuisine"])  # 去除id,cuisine列
    Y = entries.apply(lambda x: x.sum())
    ax = fig.add_subplot(5, 1, i)
    plt.plot(X, Y)
    plt.title(name)
    i = i+1

plt.tight_layout()
plt.show()

# 统计标签的分布
label_counts = data_train['cuisine'].value_counts()

# 可视化标签的分布
plt.figure(figsize=(12, 6))
sns.countplot(data=data_train, x='cuisine')
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.16)
plt.title('Distribution of Labels')
plt.show()

# 选择数量top3的标签中的一个
top3_labels = label_counts[:3].index.tolist()
selected_label = top3_labels[2]

# 筛选出所选标签的样本
selected_data = data_train[data_train['cuisine'] == selected_label]

# 统计食材的分布
ingredient_counts = selected_data.iloc[:, 2:].sum().sort_values(ascending=False)

# 可视化食材的分布
plt.figure(figsize=(12, 6))
sns.barplot(x=ingredient_counts.index[:10], y=ingredient_counts.values[:10])
plt.xticks(rotation=90)
plt.xlabel('Ingredient')
plt.ylabel('Count')
plt.gcf().subplots_adjust(bottom=0.22)
plt.title(f'Ingredient Distribution for "{selected_label}"')
plt.show()

# 将标签编码为数值
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data_train['cuisine'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 进行PCA降维
pca = PCA(n_components=2)
pca_features = pca.fit_transform(data_train.iloc[:, 2:])

# 可视化PCA结果
plt.figure(figsize=(12, 9), dpi=80)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=encoded_labels, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Analysis')
# 设置颜色标签
cbar = plt.colorbar()
cbar.set_ticks(sorted(list(label_mapping.values())))
cbar.set_ticklabels(list(label_mapping.keys()))
cbar.set_label('Cuisine')
plt.show()