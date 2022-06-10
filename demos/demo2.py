import matplotlib.pyplot as plt

# # 条形图
# Example1: 简单垂直条形图
GDP = [12406.8, 13908.57, 9386.87, 9143.64]

# 绘图
plt.bar(range(4), GDP, align='center', color='steelblue', alpha=0.8)
# Y轴标签
plt.ylabel('GDP')
# 标题
plt.title('直辖市GDP')
# X轴刻度
plt.xticks(range(4), ['北京', '上海', '天津', '重庆'])
# Y轴刻度范围
plt.ylim([5000, 15000])

# 给条形图加数值
for x, y in enumerate(GDP):
    plt.text(x, y+100, '%s' % round(y, 1), ha='center')

# 中文乱码
plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.show()


# Example2: 水平条形图
price = [34.5, 39, 32.1, 44, 28.8]

# 中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘图
plt.barh(range(5), price, align='center', color='steelblue', alpha=0.8)
# X轴标签
plt.xlabel('价格')
# 标题
plt.title('各个平台价格对比')
# Y轴刻度标签
plt.yticks(range(5), ['亚马逊', '当当', '京东', '淘宝', '天猫'])
# X轴刻度范围
plt.xlim([25, 45])

for x, y in enumerate(price):
    plt.text(y+0.1, x, '%s'%y, va='center')

plt.show()# Example2: 水平条形图
price = [34.5, 39, 32.1, 44, 28.8]

# 中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘图
plt.barh(range(5), price, align='center', color='steelblue', alpha=0.8)
# X轴标签
plt.xlabel('价格')
# 标题
plt.title('各个平台价格对比')
# Y轴刻度标签
plt.yticks(range(5), ['亚马逊', '当当', '京东', '淘宝', '天猫'])
# X轴刻度范围
plt.xlim([25, 45])

for x, y in enumerate(price):
    plt.text(y+0.1, x, '%s'%y, va='center')

plt.show()

# Example3: 组合条形图
import numpy as np
Y2016 = [15600, 12700, 11300, 4270, 3620]
Y2017 = [17400, 14800, 12000, 5200, 4020]
labels = ['北京', '上海', '香港', '深圳', '广州']
bar_width = 0.35

# 中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘图
plt.bar(np.arange(5), Y2016, label='2016', color='steelblue', alpha=0.8, width=bar_width)
plt.bar(np.arange(5)+bar_width, Y2017, label='2017', color='indianred', alpha=0.8, width=bar_width)

# 设置轴标签
plt.xlabel('Top5城市')
plt.ylabel('家庭数量')
plt.title('亿万富豪家庭数量')
# 添加X轴刻度标签
plt.xticks(np.arange(5)+bar_width, labels)
plt.ylim([2500, 19000])

# 添加数值标签
for x2016, y2016 in enumerate(Y2016):
    plt.text(x2016, y2016+100, '%s' %y2016)

for x2017, y2017 in enumerate(Y2017):
    plt.text(x2017+bar_width, y2017+100, '%s' %y2017)

# 图例
plt.legend()

plt.show()


# # # 饼图
# 中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

# 构造数据
edu = [0.2515, 0.3724, 0.3336, 0.0368, 0.0057]
labels = ['中专', '大专', '本科', '硕士', '其他']

# 突出显示大专
explode = [0, 0.1, 0, 0, 0]
# 自定义颜色
colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']
# 正圆
plt.axes(aspect='equal')

# X轴Y轴范围
plt.xlim(0, 4)
plt.ylim(0, 4)

plt.pie(
    x=edu,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%.1f%%',
    pctdistance=0.8,
    labeldistance=1.15,
    startangle=180,
    radius=1.5,
    counterclock=False,
    wedgeprops={'linewidth': 1.5, 'edgecolor':'green'},
    textprops={'fontsize':12, 'color':'k'},
    center=(1.8, 1.8),
    frame=1
)
plt.xticks(())
plt.yticks(())

plt.title('失信用户学历分布情况')

plt.show()


# # # 盒须图
import pandas as pd

path = "../data/matplotlib/train.csv"

# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv(path)
any(train.Age.isnull())
train.dropna(subset=['Age'], inplace=True)

# 主题风格
plt.style.use('ggplot')

plt.boxplot(
    x=train.Age,
    patch_artist=True,
    boxprops={'color': 'black', 'facecolor': '#9999ff'},
    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
    meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
    medianprops={'linestyle': '--', 'color': 'orange'}
)
# Y轴范围
plt.ylim(0, 85)
# 去除盒须图标签
plt.tick_params(top='off', right='off')
plt.show()

train.sort_values(by='Pclass', inplace=True)
Age = []
Levels = train.Pclass.unique()

for Pclass in Levels:
    Age.append(train.loc[train.Pclass == Pclass, 'Age'])

plt.boxplot(
    x=Age,
    labels=['一等舱', '二等舱', '三等舱'],
    patch_artist=True,
    boxprops={'color': 'black', 'facecolor': '#9999ff'},
    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
    meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
    medianprops={'linestyle': '--', 'color': 'orange'}
)
plt.show()

# # 直方图
# Example1: 简单直方图
import pandas as pd

# 乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

path = "../data/matplotlib/train.csv"
train = pd.read_csv(path)

# 删除异常数据：缺失年龄的数据
train.dropna(subset=['Age'], inplace=True)

plt.style.use('ggplot')
# 绘图：乘客年龄的频数直方图
plt.hist(train.Age,
         bins=20,
         color='steelblue',
         edgecolor='k',
         label='直方图')

# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')
# 显示图例
plt.legend()
plt.show()

# Example2: 组合直方图
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from scipy.stats import norm

# 乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

path = "../data/matplotlib/train.csv"
train = pd.read_csv(path)

plt.hist(train.Age,
         bins=np.arange(train.Age.min(), train.Age.max(), 5),
         density=True,
         color='steelblue',
         edgecolor='k')

plt.title('乘客年龄直方图')
plt.xlabel('年龄')
plt.ylabel('频率')

# 正态曲线
x1 = np.linspace(train.Age.min(), train.Age.max(), 1000)
normal = norm.pdf(x1, train.Age.mean(), train.Age.std())
line, = plt.plot(x1, normal, 'r-', linewidth=2)

plt.tick_params(top='off', right='off')

# 显示图例
plt.legend([line], ['正态分布曲线'], loc='best')

plt.show()

# # 折线图
# Example1: 简单折线图
import pandas as pd
import matplotlib.pyplot as plt
# 设置图形大小
plt.style.use('ggplot')
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

article_reading = pd.read_csv('../data/matplotlib/wechat.csv')
article_reading.date = pd.to_datetime(article_reading.date)

sub_data = article_reading.loc[article_reading.date >= '2017-08-01', :]

# 设置图形大小
fig = plt.figure(figsize=(10, 6))
plt.plot(sub_data.date,
         sub_data.article_reading_cnts,
         linestyle='-',
         linewidth=2,
         color='steelblue',
         marker='o',
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='brown')

plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

plt.tick_params(top='off', right='off')
# X轴刻度倾斜45°
fig.autofmt_xdate(rotation=45)
plt.show()

# Example2: 多元折线图
import pandas as pd
import matplotlib as mpl

# 设置图形大小
plt.style.use('ggplot')
# 解决中文乱码
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

article_reading = pd.read_csv('../data/matplotlib/wechat.csv')
article_reading.date = pd.to_datetime(article_reading.date)

sub_data = article_reading.loc[article_reading.date >= '2017-08-01', :]
fig = plt.figure(figsize=(10, 6))

# 阅读人数趋势
plt.plot(sub_data.date,
         sub_data.article_reading_cnts,
         linestyle='-',
         linewidth=2,
         color='steelblue',
         marker='o',
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='steelblue',
         label='阅读人数')

# 阅读人次趋势
plt.plot(sub_data.date,
         sub_data.article_reading_times,
         linestyle='-',
         linewidth=2,
         color='#ff9999',
         marker='o',
         markersize=6,
         markeredgecolor='black',
         markerfacecolor='#ff9999',
         label='阅读人次')

# 添加标题和坐标轴标签
plt.title('公众号每天阅读人数和人次趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

plt.tick_params(top='off', right='off')

# 图的坐标信息
ax = plt.gca()

# 设置日期的格式
date_format = mpl.dates.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(date_format)

# 设置x轴刻度的间隔天数
xlocator = mpl.ticker.MultipleLocator(3)
ax.xaxis.set_major_locator(xlocator)

fig.autofmt_xdate(rotation=45)
plt.legend()
plt.show()

# # 雷达图
import numpy as np

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

# 构造数据
values = [3.2, 2.1, 3.5, 2.8, 3]
feature = ['个人能力', 'QC知识', '解决问题能力', '服务质量意识', '团队精神']

N = len(values)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)

values = np.concatenate((values, [values[0]]))
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((feature, [feature[0]]))

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# 绘制折线图
ax.plot(angles, values, 'o-', linewidth=2)

# 填充颜色
ax.fill(angles, values, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_ylim(0, 5)
plt.title('活动前后员工状态表现')
ax.grid(True)
plt.show()

