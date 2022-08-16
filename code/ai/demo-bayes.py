"""贝叶斯模型做电商评论分类
"""
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# pd.describe_option()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

comments = pd.read_excel(r'../data/bayes/Contents.xlsx', sheet_name=0)
# print(comments.head(10))

# 剔除评论里的英文/数字/\n等字符
comments.Content = comments.Content.str.replace('[0-9a-zA-Z.\n]', '')
# print(comments.head(10))

# 加载自定义词库
jieba.load_userdict(r'../data/bayes/all_words.txt')

# 加载停止词
with open(r'../../data/bayes/mystopwords.txt', encoding='UTF-8') as words:
    stop_words = [i.strip() for i in words.readlines()]


# print(stop_words)

# 构造切词函数，并在切词时删除停止词
def cut_word(sentence):
    words = [i for i in jieba.lcut(sentence) if i not in stop_words]
    return ' '.join(words)


# 对评论内容进行切词
words = comments.Content.apply(cut_word)
# print(words[:10])

# CountVectorizer:
# 常见的特征数值计算类，是一个文本特征提取方法
# 对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率
counts = CountVectorizer(min_df=0.01)

# 词条矩阵
dtm_counts = counts.fit_transform(words).toarray()
columns = counts.get_feature_names()

X = pd.DataFrame(dtm_counts, columns=columns)
# print(X.head(10))

# 情感标签
Y = comments.Type

# 将数据拆分为训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=1)

# 构建伯努利贝叶斯分类器
bnb = naive_bayes.BernoulliNB()

# 训练拟合
bnb.fit(x_train, y_train)

# 测试数据集预测
bnb_pred = bnb.predict(x_test)

# 构建混淆矩阵
# cm = pd.crosstab(bnb_pred, y_test)
# print(cm.head(10))
print('模型的准确率为：\n', metrics.accuracy_score(y_test, bnb_pred))
print('模型的评估报告：\n', metrics.classification_report(y_test, bnb_pred))






