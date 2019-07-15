import scipy.io as scio
import pandas as pd
import numpy as np
# 导入数据
data = scio.loadmat(r"D:\works\sklearn\sklearn_book\数据集\mnist-original.mat")

x, y = data['data'].T, data['label'].T
import matplotlib.pyplot as plt
some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)

# 画一下这个图
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
# plt.axis('off') # 隐藏坐标系
# plt.show()

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# 随机拿数据
shuffle_index = np.random.permutation(60000) # permutation就是有返回值的shuffle，随机打乱
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5) # 只有"True"和"False"的array
y_test_5 = (y_test == 5)

# SGD是随机梯度下降分类，好东西
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 以下操作为了说明：交叉验证返回的是评估的分数，有时很不靠谱
# # 交叉验证
# from sklearn.model_selection import cross_val_score
# # 这样验证精度在95%左右，因为'5'只占了10%，即使全部猜不是5，也有90%以上的精度，因此不靠谱
# cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring='accuracy')

# 掏出与混淆矩阵配合的容器
from sklearn.model_selection import cross_val_predict
# 上面的交叉验证拿到的是一个精度，这里拿到的容器是一个纯净的预测，有对错的数目
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 掏出混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
# out----->	array([[53272, 1307],
# 		          [ 1077, 4344]])

# 看一下当前模型的各个参数
from sklern.precision_score, recall_score, f1_score
precision_score(y_train_5, y_train_pred)	# 准确率
recall_score(y_train_5, y_train_pred)	# 召回率
f1_score(y_train_5, y_train_pred)	# F1（准确率与召回率的折衷平均）

# 看一下上面取的那个样例的分值
y_scores = sgd_clf.decision_function([some_digit])
y_scores	# --->[161855.745]：这个数就是这个样例的分值，可以理解为于阈值的距离

# 换个容器
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 画图，x为阈值thresholds,y为准确率和召回率的曲线
plt.figure(figsize=(20, 8), dpi=40)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'r-', label='Recall')
    plt.xlabel('Threshold', fontsize=20, color='g')
    plt.xticks(color='g', fontsize=20)
    plt.yticks(color='g', fontsize=20)
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# 画一个x为召回率，y为准确率的图
plt.plot(recalls, precisions)
plt.show()


# # 训模型
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
# # 拿出预测
# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# y_train_pred_90 = (y_scores > 70000)
# y_train_pred_90
