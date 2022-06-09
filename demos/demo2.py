import matplotlib.pyplot as plt

# # 条形图
# # Example1: 简单垂直条形图
# GDP = [12406.8, 13908.57, 9386.87, 9143.64]
#
# # 绘图
# plt.bar(range(4), GDP, align='center', color='steelblue', alpha=0.8)
# # Y轴标签
# plt.ylabel('GDP')
# # 标题
# plt.title('直辖市GDP')
# # X轴刻度
# plt.xticks(range(4), ['北京', '上海', '天津', '重庆'])
# # Y轴刻度范围
# plt.ylim([5000, 15000])
#
# # 给条形图加数值
# for x, y in enumerate(GDP):
#     plt.text(x, y+100, '%s' % round(y, 1), ha='center')
#
# # 中文乱码
# plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.show()


# # Example2: 水平条形图
# price = [34.5, 39, 32.1, 44, 28.8]
#
# # 中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 绘图
# plt.barh(range(5), price, align='center', color='steelblue', alpha=0.8)
# # X轴标签
# plt.xlabel('价格')
# # 标题
# plt.title('各个平台价格对比')
# # Y轴刻度标签
# plt.yticks(range(5), ['亚马逊', '当当', '京东', '淘宝', '天猫'])
# # X轴刻度范围
# plt.xlim([25, 45])
#
# for x, y in enumerate(price):
#     plt.text(y+0.1, x, '%s'%y, va='center')
#
# plt.show()# Example2: 水平条形图
# price = [34.5, 39, 32.1, 44, 28.8]
#
# # 中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 绘图
# plt.barh(range(5), price, align='center', color='steelblue', alpha=0.8)
# # X轴标签
# plt.xlabel('价格')
# # 标题
# plt.title('各个平台价格对比')
# # Y轴刻度标签
# plt.yticks(range(5), ['亚马逊', '当当', '京东', '淘宝', '天猫'])
# # X轴刻度范围
# plt.xlim([25, 45])
#
# for x, y in enumerate(price):
#     plt.text(y+0.1, x, '%s'%y, va='center')
#
# plt.show()

# # Example3: 组合条形图
# import numpy as np
# Y2016 = [15600, 12700, 11300, 4270, 3620]
# Y2017 = [17400, 14800, 12000, 5200, 4020]
# labels = ['北京', '上海', '香港', '深圳', '广州']
# bar_width = 0.35
#
# # 中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 绘图
# plt.bar(np.arange(5), Y2016, label='2016', color='steelblue', alpha=0.8, width=bar_width)
# plt.bar(np.arange(5)+bar_width, Y2017, label='2017', color='indianred', alpha=0.8, width=bar_width)
#
# # 设置轴标签
# plt.xlabel('Top5城市')
# plt.ylabel('家庭数量')
# plt.title('亿万富豪家庭数量')
# # 添加X轴刻度标签
# plt.xticks(np.arange(5)+bar_width, labels)
# plt.ylim([2500, 19000])
#
# # 添加数值标签
# for x2016, y2016 in enumerate(Y2016):
#     plt.text(x2016, y2016+100, '%s' %y2016)
#
# for x2017, y2017 in enumerate(Y2017):
#     plt.text(x2017+bar_width, y2017+100, '%s' %y2017)
#
# # 图例
# plt.legend()
#
# plt.show()


# # # 饼图
# # 中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 设置风格
# plt.style.use('ggplot')
#
# # 构造数据
# edu = [0.2515, 0.3724, 0.3336, 0.0368, 0.0057]
# labels = ['中专', '大专', '本科', '硕士', '其他']
#
# # 突出显示大专
# explode = [0, 0.1, 0, 0, 0]
# # 自定义颜色
# colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']
# # 正圆
# plt.axes(aspect='equal')
#
# # X轴Y轴范围
# plt.xlim(0, 4)
# plt.ylim(0, 4)
#
# plt.pie(
#     x=edu,
#     explode=explode,
#     labels=labels,
#     colors=colors,
#     autopct='%.1f%%',
#     pctdistance=0.8,
#     labeldistance=1.15,
#     startangle=180,
#     radius=1.5,
#     counterclock=False,
#     wedgeprops={'linewidth': 1.5, 'edgecolor':'green'},
#     textprops={'fontsize':12, 'color':'k'},
#     center=(1.8, 1.8),
#     frame=1
# )
# plt.xticks(())
# plt.yticks(())
#
# plt.title('失信用户学历分布情况')
#
# plt.show()


# # # 盒须图
# import pandas as pd
#
# path = "../data/matplotlib/train.csv"
#
# # 解决中文乱码
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False
#
# train = pd.read_csv(path)
# any(train.Age.isnull())
# train.dropna(subset=['Age'], inplace=True)
#
# # 主题风格
# plt.style.use('ggplot')
#
# plt.boxplot(
#     x=train.Age,
#     patch_artist=True,
#     boxprops={'color': 'black', 'facecolor': '#9999ff'},
#     flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
#     meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
#     medianprops={'linestyle': '--', 'color': 'orange'}
# )
# # Y轴范围
# plt.ylim(0, 85)
# # 去除盒须图标签
# plt.tick_params(top='off', right='off')
# plt.show()
#
# train.sort_values(by='Pclass', inplace=True)
# Age = []
# Levels = train.Pclass.unique()
#
# for Pclass in Levels:
#     Age.append(train.loc[train.Pclass == Pclass, 'Age'])
#
# plt.boxplot(
#     x=Age,
#     labels=['一等舱', '二等舱', '三等舱'],
#     patch_artist=True,
#     boxprops={'color': 'black', 'facecolor': '#9999ff'},
#     flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
#     meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
#     medianprops={'linestyle': '--', 'color': 'orange'}
# )
# plt.show()

# # 直方图
# Example1
import numpy as np
import pandas as pd

# 乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False






