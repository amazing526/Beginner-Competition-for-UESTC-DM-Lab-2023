import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

data_train = pd.read_csv("../data/recipes_train.csv")
data_train = data_train.drop(columns=["id"])
data_train['cuisine'] = data_train['cuisine'].map({'chinese': 0, 'indian': 1, 'japanese': 2, 'korean': 3, 'thai': 4})
data_test = pd.read_csv("../data/recipes_test.csv")

data_train_X = data_train.drop(columns=["cuisine"]).values
data_train_y = data_train["cuisine"].values

X_train, X_valid, y_train, y_valid = train_test_split(data_train_X, data_train_y, test_size=0.2, random_state=526)

# 选择模型
# Keras有两种类型的模型，序贯模型（Sequential）和函数式模型（Model），函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况。
# a）序贯模型（Sequential):单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接。这种模型编译速度快，操作也比较简单
# b）函数式模型（Model）：多输入多输出，层与层之间任意连接。这种模型编译速度慢。
mlp = Sequential([
    Dense(100, activation='relu', input_dim=383),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                   min_delta=0.01,
                   patience=10,
                   verbose=2, mode='min')

mlp.fit(X_train, y_train, epochs=50, callbacks=[es], validation_data=(X_valid, y_valid))

_, train_accuracy = mlp.evaluate(X_train, y_train)
_, valid_accuracy = mlp.evaluate(X_valid, y_valid)

print('train accuracy: {0:0.2f}'.format(train_accuracy))
print('valid accuracy: {0:0.2f}'.format(valid_accuracy))

y_valid = label_binarize(y_valid, classes=[0, 1, 2, 3, 4])
y_score = mlp.predict_proba(X_valid)
precision = dict()
recall = dict()
average_precision = dict()
for i in range(5):
    precision[i], recall[i], _ = precision_recall_curve(y_valid[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_valid[:, i], y_score[:, i])

# A "macro-average": quantifying score on all classes jointly
precision["macro"], recall["macro"], _ = precision_recall_curve(y_valid.ravel(), y_score.ravel())
average_precision["macro"] = average_precision_score(y_valid, y_score, average="macro")

plt.step(recall['macro'], precision['macro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, macro-averaged over all classes: AP={0:0.2f}'.format(average_precision["macro"]))
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 1
colors = ['blue', 'red', 'green', 'black', 'yellow']
classes = ['chinese', 'indian', 'japanese', 'korean', 'thai']
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.3f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

# 输入训练集x的值对y进行预测
result = mlp.predict(data_test.drop(columns=["id"]).values)
# 找出0~4五个概率之中最大的那一个数
result = np.argmax(result, axis=1)

output_df = pd.DataFrame()
output_df["id"] = data_test["id"]
output_df["cuisine"] = result
output_df['cuisine'] = output_df['cuisine'].replace({0: 'chinese', 1: 'indian', 2: 'japanese', 3: 'korean', 4: 'thai'})

output_df.to_csv("../result/submission_MLP.csv", index=False)
