import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1.读取数据
df = pd.read_excel('../../../data/ai/客户价值数据表.xlsx')
df.head()  # 显示前5行数据

X = df[['历史贷款金额', '贷款次数', '学历', '月收入', '性别']]
Y = df['客户价值']

# 2.模型搭建
model = LinearRegression()
model.fit(X, Y)

# 3.线性回归方程构造
print('各系数为:' + str(model.coef_))
print('常数项系数k0为:' + str(model.intercept_))

# 4.模型评估
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2).fit()
est.summary()
