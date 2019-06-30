import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取文件
data = pd.read_csv(r'datasets\housing\housing.csv')
# 换列名
data.columns = ['经度',  '维度', '房屋年龄中位数', '总房间数', '总卧室数', '人口数', '家庭数'
               , '收入中位数', '房屋价值中位数', '与大海的距离']
# 切分训练集和测试集，后面好像没有用到
train_set, test_set = train_test_split(data, test_size=0.2, random_state=666)

# 缩放特征，将连续性数据打成离散型时范围会小一些，
data['income'] = np.ceil(data['收入中位数'] / 1.5)
data['income'].where(data['income'] < 5, 5, inplace=True)
# 分层切分，比上述切分优秀
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=666)
for train_index, test_index in split.split(data, data['income']): # income列存在的意义就是提供我分层切片的标准，后面还要把income列删掉
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
# 在训练集和测试集删除income
strat_train_set.drop('income', axis=1, inplace=True)
strat_test_set.drop('income', axis=1, inplace=True)
# 复制一份训练集
housing = strat_train_set.copy()

# 画图
# plt.figure(figsize=(20, 20), facecolor='b', dpi=128)
# housing.plot(kind="scatter", x="经度", y="维度", alpha=0.4,
#     s=housing["人口数"]/100, label="人口数",
#     c="房屋价值中位数", cmap=plt.get_cmap("jet"), colorbar=True,
# )
# plt.legend()

# 查看相关系数
corr_matrix = housing.corr()
corr_matrix['房屋价值中位数'].sort_values(ascending=False)

# sns画图，看每个特征的联系
# import seaborn as sns
# sns.scatterplot(x='收入中位数', y='房屋价值中位数', data=housing, alpha=0.3)

# 造特征
housing['每户房间数'] = housing['总房间数'] / housing['家庭数']
housing['每个房间卧室数'] = housing['总卧室数'] / housing['总房间数']
housing['每家人口数'] = housing['人口数'] / housing['家庭数']

# 这两个就是用于训练的样本和标签
housing_data = housing.drop('房屋价值中位数', axis=1)
housing_labels = housing['房屋价值中位数'].copy()

# 处理null值
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
housing_num = housing_data.drop('与大海的距离', axis=1)
imputer.fit(housing_num)    # 拿出估计器，以每一列中位数为基准
x = imputer.transform(housing_num) # 拿出转换器
housing_tr = pd.DataFrame(x, columns=housing_num.columns) # 转成df，这个现在是训练集

# 将文本列换为独热向量
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer(sparse_output=True)
housing_cat = housing['与大海的距离']
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray().shape

# 将独热向量转成df的格式，向量有几维就是几个特征
housing_hot = pd.DataFrame(housing_cat_1hot.toarray(), columns=encoder.classes_)
# 拼接数据拿到最终的训练集
housing_prepared = pd.concat([housing_tr, housing_hot], axis=1)

# 训练模型
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

# 按照特征的重要程度排序，踢掉不重要的特征
feature_importances = forest_reg.feature_importances_
sorted(zip(housing_prepared.columns, feature_importances), key=lambda x: x[1], reverse=True)