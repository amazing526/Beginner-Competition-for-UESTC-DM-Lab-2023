import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import BaggingClassifier

data_train = pd.read_csv("../data/recipes_train.csv")
data_train = data_train.drop(columns=["id"])
data_test = pd.read_csv("../data/recipes_test.csv")
data_train_X = data_train.drop(columns=["cuisine"]).values
data_train_y = data_train["cuisine"].values
X_train, X_valid, y_train, y_valid = train_test_split(data_train_X, data_train_y, test_size=0.2, random_state=526)

sgd = SGDClassifier(alpha=0.0001, max_iter=1000, loss='log', random_state=526)
clf = BaggingClassifier(base_estimator=sgd, n_estimators=10, max_samples=1.0, max_features=1.0,
                        bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=526)
clf.fit(X_train, y_train)

print('train accuracy: {0:0.2f}'.format(clf.score(X_train, y_train)))
print('valid accuracy: {0:0.2f}'.format(clf.score(X_valid, y_valid)))

classes = ['chinese', 'indian', 'japanese', 'korean', 'thai']
y_valid = label_binarize(y_valid, classes=classes)
y_score = clf.predict_proba(X_valid)
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
plt.ylim([0.0, 1.05])
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
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

result = clf.predict(data_test.drop(columns=["id"]).values)
output_df = pd.DataFrame()
output_df["id"] = data_test["id"]
output_df["cuisine"] = result
output_df.to_csv("../result/submission_integration.csv", index=False)