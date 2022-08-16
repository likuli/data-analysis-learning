from collections import OrderedDict
import pandas as pd

examDict = {
    '学习时间': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25,
             2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
    '分数': [10, 22, 13, 43, 20, 22, 33, 50, 62,
           48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]
}
examOrderDict = OrderedDict(examDict)
exam = pd.DataFrame(examOrderDict)

# 查看数据格式
# print(exam.head())

import matplotlib.pyplot as plt

exam_X = exam['学习时间']
exam_Y = exam['分数']

plt.scatter(exam_X, exam_Y, color = 'green')
plt.ylabel('scores')
plt.xlabel('times')
plt.title('exam data')
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(exam_X,
                                                    exam_Y,
                                                    train_size = 0.8)
# 查看分割后的结果
# print(X_train.head())
# print(X_train.shape)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
#从skl中导入线性回归的模型
from sklearn.linear_model import LinearRegression
#创建一个模型
model = LinearRegression()
#训练一下
model.fit(X_train, Y_train)

print(model.predict([[1.5]]))

import matplotlib.pyplot as plt
#绘制散点图
plt.scatter(exam_X, exam_Y, color = 'green', label = 'train data')
#设定X,Y轴标签和title
plt.ylabel('scores')
plt.xlabel('times')

#绘制最佳拟合曲线
Y_train_pred = model.predict(X_train)
plt.plot(X_train, Y_train_pred, color = 'black', label = 'best line')

#来个图例
plt.legend(loc = 2)

# plt.show()

a = model.intercept_
b = model.coef_
a = float(a)
b = float(b)
print('该模型的线性回归方程为：y = {} + {} * x'.format(a, b))

