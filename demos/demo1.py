import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import pandas as pd

# Example1
path1 = "../data/pandas/chipotle.tsv"

# step1: 加载数据
chipo = pd.read_csv(path1, sep="\t")

# step2: 查看数据前10行
print(chipo.head(10))

# step3: 查看数据有多少列
print(chipo.shape[1])

# step4: 打印全部列名
print(chipo.columns)

# step5: 查看数据集索引
print(chipo.index)

# step6: 查看下单数量最多的商品
c = chipo[['item_name', 'quantity']].groupby(['item_name'], as_index=False).agg({'quantity': sum})
c.sort_values(['quantity'], ascending=False, inplace=True)
print(c.head())

# step7: 查看有多少种商品
print(chipo['item_name'].nunique())

# step8: 在choice_description中，下单次数最多的商品是什么
print(chipo['choice_description'].value_counts().head())

# step9: 下单商品总量
print(chipo['quantity'].sum())

# step10: 将价格item_price转为浮点数
d = lambda x: float(x[1: -1])
chipo['item_price'] = chipo['item_price'].apply(d)

# step11: 计算总收入
chipo['sub_total'] = round(chipo['item_price'] * chipo['quantity'], 2)
print(chipo['sub_total'].sum())

# step12: 订单总量
print(chipo['order_id'].nunique())


# Example2: 数据过滤与排序
path2 = "../data/pandas/Euro2012_stats.csv"

# step1: 加载数据
euro = pd.read_csv(path2)

# step2: 读取Goals列
print(euro.Goals)

# step3: 统计球队数量
print(euro.shape[0])

# step4: 查看数据集信息
print(euro.info())

# step5: 将Team、Yellow Cards、Red Cards单独存储到一个数据集
discipline = euro[['Team', 'Yellow Cards', 'Red Cards']]
# print(discipline)

# step6: 对数据集discipline按Red Cards、Yellow Cards排序
print(discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False))

# step7: 计算黄牌平均值
print(round(discipline['Yellow Cards'].mean()))

# step8: 找出进球数大于6的球队
print(euro[euro.Goals > 6])

# step9: 选取G开头的球队
print(euro[euro.Team.str.startswith('G')])

# step10: 选取前7列
print(euro.iloc[:, 0:7])

# step11: 选取除了最后3列之外的全部列
print(euro.iloc[:, :-3])

# step12: 找到英格兰(England)、意大利(Italy)和俄罗斯(Russia)的射正率(Shooting Accuracy)
print(euro.loc[euro.Team.isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']])


# Example3: 数据分组
path3 = "../data/pandas/drinks.csv"

# step1: 载入数据
drinks = pd.read_csv(path3)
print(drinks.head())

# step2: 计算各大洲啤酒平均消耗量
print(drinks.groupby('continent').beer_servings.mean())

# step3: 计算各大洲红酒平均消耗量
print(drinks.groupby('continent').wine_servings.mean())

# step4: 打印出各大洲每种酒类别的消耗平均值
print(drinks.groupby('continent').mean())

# step4: 打印出各大洲每种酒类别的消耗中位数
print(drinks.groupby('continent').median())

# step5:  打印出各大洲对spirit饮品消耗的平均值，最大值和最小值
print(drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max']))


# Example4
path4 = "../data/pandas/US_Crime_Rates_1960_2014.csv"

# step1: 加载数据
crime = pd.read_csv(path4)
print(crime.head())

# step2: 查看数据集信息
print(crime.info())

# step3: 将Year列数据类型转为datetime64
crime.Year = pd.to_datetime(crime.Year, format='%Y')
print(crime.info())

# step4: 将Year设置为数据集索引
crime = crime.set_index('Year', drop=True)
print(crime.head())

# step5: 删除Total列
del crime['Total']
print(crime.head())

# step6: 按照Year对数据进行分组求和
crimes = crime.resample('10AS').sum()
population = crime['Population'].resample('10AS').max()
crimes['Population'] = population
print(crimes)

# step7: 打印历史最危险的时代
print(crime.idxmax())


# Example5
# step1: 构造测试数据
raw_data_1 = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

# step2: 装载数据
data1 = pd.DataFrame(raw_data_1, columns=['subject_id', 'first_name', 'last_name'])
data2 = pd.DataFrame(raw_data_2, columns=['subject_id', 'first_name', 'last_name'])
data3 = pd.DataFrame(raw_data_3, columns=['subject_id', 'test_id'])

# step3: 行维度合并data1、data2
all_data = pd.concat([data1, data2])
print(all_data)

# step4: 列维度合并data1、data2
all_data_col = pd.concat([data1, data2], axis=1)
print(all_data_col)

# step5: 按照subject_id，合并data_all和data3
print(pd.merge(all_data, data3, on='subject_id'))

# step6: 按照subject_id，合并data1、data2
print(pd.merge(data1, data2, on='subject_id', how='inner'))

# step7: 按照subject_id，合并data1、data2
print(pd.merge(data1, data2, on='subject_id', how='outer'))


# Example6
import datetime
path6 = "../data/pandas/wind.data"

# step1: 加载数据,并设置前三列为合适的索引
data = pd.read_table(path6, sep="\s+", parse_dates=[[0, 1, 2]])
print(data.head())

# step2:  修复step1中自动创建索引的错误数据(2061年？)
def fix_year(x):
    year = x.year - 100 if x.year > 1989 else x.year
    return datetime.date(year, x.month, x.day)
data['Yr_Mo_Dy'] = data['Yr_Mo_Dy'].apply(fix_year)
print(data.head())

# step3: 将Yr_Mo_Dy设置为索引，类型datetime64[ns]
data['Yr_Mo_Dy'] = pd.to_datetime(data['Yr_Mo_Dy'])
data = data.set_index('Yr_Mo_Dy')
print(data.head())

# step4: 统计每个location数据缺失值
print(data.isnull().sum())

# step5: 统计每个location数据完整值
print(data.shape[0] - data.isnull().sum())

# step6: 计算所有数据平均值
print(data.mean().mean())

# step7: 创建数据集，存储每个location最小值、最大值、平均值、标准差
loc_stats = pd.DataFrame()
loc_stats['min'] = data.min()
loc_stats['max'] = data.max()
loc_stats['mean'] = data.mean()
loc_stats['std'] = data.std()
print(loc_stats)

# step8: 创建数据集，存储所有location最小值、最大值、平均值、标准差
day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)
print(day_stats.head())


# Example7: 泰坦尼克灾难数据
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
path7 = "../data/pandas/train.csv"

# step1: 加载数据
titanic = pd.read_csv(path7)
print(titanic.head())

# step2: 设置索引
titanic = titanic.set_index('PassengerId')
print(titanic.head())

# step3: 分别统计男女乘客数量
male_sum = (titanic['Sex'] == 'male').sum()
female_sum = (titanic['Sex'] == 'female').sum()
print(male_sum, female_sum)

# step4: 绘制表示乘客票价、年龄、性别的散点图
lm = sns.lmplot(x='Age', y='Fare', data=titanic, hue='Sex', fit_reg=False)
lm.set(title='Fare x Age')
axes = lm.axes
axes[0, 0].set_ylim(-5,)
axes[0, 0].set_xlim(-5, 85)
plt.show()

# step5: 统计生还人数
print(titanic.Survived.sum())

# step6: 绘制展示票价的直方图
df = titanic.Fare.sort_values(ascending=False)
print(df)
binsVal = np.arange(0, 600, 10)
print(binsVal)

plt.hist(df, bins=binsVal)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare Payed Histrogram')
plt.show()


# Example8
# step1: 构造数据
raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']
            }
pokemon = pd.DataFrame(raw_data)
print(pokemon.head())

# step2: 修改列排序
pokemon = pokemon[['name', 'type', 'hp', 'evolution','pokedex']]
print(pokemon.head())

# step3: 新增place列
pokemon['place'] = ['park','street','lake','forest']
print(pokemon.head())

# step4: 查看每列的数据类型
print(pokemon.dtypes)


# Example9: Apple公司股价数据
path9 = "../data/pandas/Apple_stock.csv"

# step1: 加载数据
apple = pd.read_csv(path9)
print(apple.head())

# step2: 查看每列的数据类型
print(apple.dtypes)

# step3: 将Date转换为datetime类型
apple.Date = pd.to_datetime(apple.Date)
print(apple['Date'].head())

# step4: 将Date设置为索引
apple = apple.set_index('Date')
print(apple.head())

# step5: 查看是否有重复日期
print(apple.index.is_unique)

# step6: 将index设置为升序
apple = apple.sort_index(ascending=True)
print(apple.head())

# step7: 获取每月的最后一个交易日
apple_month = apple.resample('BM').last()
print(apple_month.head())

# step8: 计算数据集中最早日期和最晚日期相差多少天
print((apple.index.max() - apple.index.min()).days)

# step9: 计算数据集中一共有多少个月
apple_months = apple.resample('BM').mean()
print(apple_months.index)

# step10: 按照时间顺序可视化Adj Close值
appl_open = apple['Adj Close'].plot(title = "Apple Stock")
fig = appl_open.get_figure()
fig.set_size_inches(13.5, 9)
plt.show()


# Example10: Iris纸鸢花数据
path10 = "../data/pandas/iris.csv"

# step1: 加载数据
iris = pd.read_csv(path10)
print(iris.head())

# step2: 添加列名称
iris = pd.read_csv(path10, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(iris.head())

# step3: 查看是否有缺失值
print(iris.isnull().sum())

# step4: 将列petal_length的第10到19行设置为缺失值
iris.iloc[10:20, 2:3] = np.nan
print(iris.head(20))

# step5: 将缺失值替换为1.0
iris.petal_length.fillna(1, inplace=True)
print(iris.head(20))

# step6: 删除class列
del iris['class']
print(iris.head())

# step7: 数据集前三行设置为NaN
iris.iloc[0:3, :] = np.nan
print(iris.head())

# step8: 删除含有NaN的行
iris = iris.dropna(how='any')
print(iris.head())

# step9: 重置索引
iris = iris.reset_index(drop=True)
print(iris.head())



