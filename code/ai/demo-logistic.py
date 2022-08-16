"""
逻辑归回Demo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# pd.describe_option()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# step1: 加载数据
data = pd.read_csv('../../data/ai/O2O/L2_Week3.csv')
# print(data.head())
# print(data.shape)

# step2: 查看数据是否有空值
# print(data.isnull().sum())

# step3: coupon_ind是要预测的目标值
# 查看目标值的分布
# print(data['coupon_ind'].value_counts(1))

# step4: 查看特征值和目标值之间的关系
# corr计算列与列之间的相关系数，返回相关系数矩阵
# print(data.corr()[['coupon_ind']])
# sns.heatmap(data.corr()[['coupon_ind']])
# plt.show()

# step5: 选取特征值
X = data.iloc[:, 0:9]
# print(X.head())

# One-Hot处理
job = pd.get_dummies(X['job'])
marital = pd.get_dummies(X['marital'])
default = pd.get_dummies(X['default'])
returned = pd.get_dummies(X['returned'])
loan = pd.get_dummies(X['loan'])

XX = pd.concat([X, job, marital, default, returned, loan], axis=1)

XXX = pd.concat([XX, data['coupon_ind']], axis=1)

# sns.heatmap(XXX.corr()[['coupon_ind']])
# plt.show()

# 查看类别型变量的所有类别及类别分布概率情况
# print(X['job'].unique())
# print(X['marital'].unique())
# print(X['default'].unique())
# print(X['returned'].unique())
# print(X['loan'].unique())

# print(data['job'].value_counts())
# print(data['marital'].value_counts())
# print(data['default'].value_counts())
# print(data['returned'].value_counts())
# print(data['loan'].value_counts())

# 将为进行独热编码的特征删除
x = [2, 3, 4, 5, 6]
XX.drop(XX.columns[x], axis=1, inplace=True)
# print(XX.head())

# 划分训练集和测试集
XX_train, XX_test, Y_train, Y_test = train_test_split(XX, data['coupon_ind'], train_size=0.75)

model = LogisticRegression(C=10000.0, class_weight='balanced', dual=False,
                           fit_intercept=True, intercept_scaling=1, max_iter=100,
                           multi_class='ovr', n_jobs=1, penalty='l2', random_state=100,
                           solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
model.fit(XX_train, Y_train)

prob = model.predict_proba(XX_train)
r = pd.DataFrame(prob).apply(lambda x: round(x, 4))

# 查看预测结果准确率
pred = model.predict(XX_test)
print(classification_report(Y_test, pred, labels=[1, 0], target_names=['是', '否']))

print('截距为:', model.intercept_)
print('回归系数为:', model.coef_)
print('训练集正确率', model.score(XX_train, Y_train))
print('预测值正确率', metrics.accuracy_score(Y_test, pred))
# metrics.

