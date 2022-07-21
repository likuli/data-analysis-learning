在掌握Numpy基础之后，为了能将Numpy快速应用于实践，特意梳理了这100道练习，供大家学习。

练习题难易程度分为四个级别，其中☆最简单，☆☆☆☆最难。

记住：**看，没用！码起来才是王道！**

练习题难度分布：

| 难度 | 题量 | 百分比 |
| ---- | ------- | ------- |
|☆ | 0 | 25%|
|☆☆| 0 | 25%|
|☆☆☆| 0 | 25% |
|☆☆☆☆ | 0 | 25% |

#### 1、查看Numpy版本    ☆
```python
import numpy as np
print(np.__version__)

# 输出：1.21.6
```

#### 2、创建一个从 0 到 9 的一维数组    ☆

```python
import numpy as np
arr = np.arange(10)

# 输出：[0 1 2 3 4 5 6 7 8 9]
```

#### 3、如何创建一个bool数组？ ☆
```python
# 创建一个值全部为True的3x3的Numpy数组
np.full((3, 3), True, dtype=bool)

# array([[ True,  True,  True],
#       [ True,  True,  True],
#       [ True,  True,  True]])
```

#### 4、从一维数组中提取满足给定条件的数字    ☆
```python
# 从数组中提取所有的奇数数字
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
odd_arr = arr[arr % 2 == 1]
print(odd_arr)

# 输出：[1 3 5 7 9]
```

#### 5、用指定值替换Numpy数组中满足条件的数值    ☆
```python
# 将数组中所有奇数替换为-1
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[arr % 2 == 1] = -1
print(arr)

# 输出：[ 0 -1  2 -1  4 -1  6 -1  8 -1]
```

#### 6、如何在不改变原始数据的情况下，替换满足指定条件的数值？ ☆☆
```python
arr = np.arange(10)
out = np.where(arr % 2 == 1, -1, arr)
print(arr)
print(out)

# 输出：
# [0 1 2 3 4 5 6 7 8 9]
# [ 0 -1  2 -1  4 -1  6 -1  8 -1]
```

#### 7、如何改变数组的形状？   ☆
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr.reshape(2, -1)

# array([[0, 1, 2, 3, 4],
#       [5, 6, 7, 8, 9]])
```

#### 8、如何垂直拼接两个数组？ ☆☆
```python
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)

# 三种实现方式
# Method 1:
np.concatenate([a, b], axis=0)

# Method 2:
np.vstack([a, b])

# Method 3:
np.r_[a, b]

# 输出：array([[0, 1, 2, 3, 4],
#       [5, 6, 7, 8, 9],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 1, 1]])
```

#### 9、如何水平拼接两个数组？  ☆☆
```python
a = np.arange(10).reshape(2, -1)
b = np.repeat(1, 10).reshape(2, -1)

# 三种实现方式
# Method 1:
np.concatenate([a, b], axis=1)

# Method 2:
np.hstack([a, b])

# Method 3:
np.c_[a, b]

# 输出 array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
#           [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
```

#### 10、如何在没有硬编码的情况下实现自定义序列？ ☆☆
```python
# 仅使用指定数组和Numpy函数完成自定义序列
a = np.array([1,2,3])
np.r_[np.repeat(a, 3), np.tile(a, 3)]

# 输出： 
# array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
```

#### 11、如何获取两个数组之间的相同数值？    ☆☆
```python
# 获取a、b两个数组的交集
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)

# 输出：array([2, 4])
```

#### 12、如何从一个数组中删除另一个数组中存在的数值？  ☆☆
```python
# 从数组a中删除所有数组b中含有的值
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)

# 输出：
# array([1, 2, 3, 4])
```

#### 13、如何获取两个数组元素匹配的位置？    ☆☆
```python
# 获取a、b两数组中元素相等的位置下标
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)

# 输出：
# (array([1, 3, 5, 7], dtype=int64),)
```

#### 14、如何从numpy数组中提取给定范围之间的所有数字？   ☆☆
```python
# 从数组a中获取所有在5~10之间的元素
a = np.arange(15)

# 三种实现方法：
# Method 1
index = np.where((a >= 5) & (a <= 10))
a[index]


# Method 2
index = np.where(np.logical_and(a>=5, a<=10))
a[index]

# Method 3
a[(a >= 5) & (a <= 10)]

# 输出：
# array([ 5,  6,  7,  8,  9, 10])
```

#### 15、如何使处理标量的python函数在numpy数组上工作？    ☆☆
```python
# 后续补充！！！！！！！！
```

#### 16、如何交换二维numpy数组中的两列？ ☆☆
```python
# 交换数组a中第1、2列的位置
a = np.arange(9).reshape(3, 3)
a[:, [1, 0, 2]]

# 输出：array([[1, 0, 2],
#            [4, 3, 5],
#            [7, 6, 8]])

```

#### 17、如何交换二维numpy数组中的两行？ ☆☆
```python
a = np.arange(9).reshape(3,3)

a[[1, 0, 2], :]

#输出： 
# array([[3, 4, 5],
#        [0, 1, 2],
#        [6, 7, 8]])
```

#### 18、如何反转二维numpy数组的行？ ☆☆
```python
a = np.arange(9).reshape(3,3)
a[::-1]

# 输出：
# array([[6, 7, 8],
#        [3, 4, 5],
#        [0, 1, 2]])

```

#### 19、如何反转二维numpy数组的列？ ☆☆
```python
a = np.arange(9).reshape(3,3)
a[:, ::-1]

# 输出：
# array([[2, 1, 0],
#        [5, 4, 3],
#        [8, 7, 6]])
```


#### 20、如何创建一个包含 5 到 10 之间的随机浮点数的二维数组？☆☆
```python
# 创建一个 5x3 的二维数组，包含 5 到 10 之间的随机十进制数。

# 两种实现方法
# Method 1:
rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
print(rand_arr)

# Method 2:
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)

# 输出：
# [[6.65021967 7.70718224 6.86337495]
#  [6.56448304 5.51846632 9.8474961 ]
#  [7.27059929 7.49226314 9.13246733]
#  [5.36998493 5.67076359 7.1329322 ]
#  [7.94925269 8.19270188 7.18788765]]
```

#### 21、如何在 numpy 数组中只打印 3 位小数？ ☆
```python
rand_arr = np.random.random([5,3])

# Limit to 3 decimal places
np.set_printoptions(precision=3)
rand_arr[:4]

# 输出：
# array([[0.274, 0.045, 0.779],
#        [0.059, 0.113, 0.144],
#        [0.365, 0.228, 0.258],
#        [0.053, 0.718, 0.867]])
```

#### 22、如同以科学计数法打印 numpy 数组（例如：1e3）？ ☆
```python
# 重置打印配置
np.set_printoptions(suppress=False)

# 创建随机数组
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr
# 输出：
# array([[  5.434049e-04,   2.783694e-04,   4.245176e-04],
#        [  8.447761e-04,   4.718856e-06,   1.215691e-04],
#        [  6.707491e-04,   8.258528e-04,   1.367066e-04]])

np.set_printoptions(suppress=True, precision=6)  
rand_arr
# 输出：
# array([[ 0.000543,  0.000278,  0.000425],
#        [ 0.000845,  0.000005,  0.000122],
#        [ 0.000671,  0.000826,  0.000137]])
```

#### 23、如何打印指定数量的元素个数？ ☆
```python
# 限制 numpy 数组打印的输出个数
np.set_printoptions(threshold=6)
arr = np.arange(15)
arr
# 输出：
# array([ 0,  1,  2, ..., 12, 13, 14])
```

#### 24、如何在不截断的情况下打印完整的 numpy 数组？ ☆
```python
np.set_printoptions(threshold=6)
arr = np.arange(15)

# Solution
np.set_printoptions(threshold=np.nan)
arr
# 输出：
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

#### 25、如何导入既包含数字又包含文本的数据集，并保持文本数据完整？ ☆☆

```python
data = np.loadtxt('./data/analysis_tool/iris.csv', delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
data[:3]

# 输出：
# array([['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
#        ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa'],
#        ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa']], dtype=object)
```

#### 26、如何从一维 numpy 数组中提取指定列？ ☆☆
```python
import numpy as np

arr = np.arange(0, 16).reshape(4, 4)
print(arr.shape)
# 输出：(4, 4)

# 提取第4列
cols = np.array([row[3] for row in arr])
cols
# 输出： 
# array([ 3,  7, 11, 15])
```

#### 27、如何将一维元组数组转换为二维 numpy 数组？ ☆☆
```python
file = '../../data/analysis_tool/iris.csv'

# 两种实现方法：
# Method 1: Convert each row to a list and get the first 4 items
iris_1d = np.genfromtxt(file, delimiter=',', dtype=None)
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
iris_2d[:4]

# Alt Method 2: Import only the first 4 columns from source data
iris_2d = np.genfromtxt(file, delimiter=',', dtype='float', usecols=[0])
iris_2d[:4]

# 输出：
# array([[5.1, 3.5, 1.4, 0.2],
#       [4.9, 3. , 1.4, 0.2],
#       [4.7, 3.2, 1.3, 0.2],
#       [4.6, 3.1, 1.5, 0.2]])

```

#### 28、如何计算 numpy 数组的均值、中值、标准差？ ☆

```python
import numpy as np
file = '../../data/analysis_tool/iris.csv'
data = np.genfromtxt(file, delimiter=',', dtype=None, usecols=[0])
mu, med, sd = np.mean(data), np.median(data), np.std(data)
print(mu, med, sd)

# 输出：5.843333333333334 5.8 0.8253012917851409
```
 
#### 29、如何规范化一个数组，使值的范围正好在 0 和 1 之间？ ☆☆
```python
# 数组归一化处理
import numpy as np
file = '../../data/analysis_tool/iris.csv'
data = np.genfromtxt(file, delimiter=',', dtype=None, usecols=[0])
Dmax, Dmin = data.max(), data.min()

# 两种解决方案：
# Method 1:
D = (data - Dmin) / (Dmax - Dmin)

# Method 2:
D = (data - Dmin) / data.ptp()

print(D)

# 输出：
# [0.222 0.167 0.111 0.083 0.194 0.306 0.083 0.194 0.028 0.167 0.306 0.139
#  0.139 0.    0.417 0.389 0.306 0.222 0.389 0.222 0.306 0.222 0.083 0.222
#  0.139 0.194 0.194 0.25  0.25  0.111 0.139 0.306 0.25  0.333 0.167 0.194
#  0.333 0.167 0.028 0.222 0.194 0.056 0.028 0.194 0.222 0.139 0.222 0.083
#  0.278 0.194 0.75  0.583 0.722 0.333 0.611 0.389 0.556 0.167 0.639 0.25
#  0.194 0.444 0.472 0.5   0.361 0.667 0.361 0.417 0.528 0.361 0.444 0.5
#  0.556 0.5   0.583 0.639 0.694 0.667 0.472 0.389 0.333 0.333 0.417 0.472
#  0.306 0.472 0.667 0.556 0.361 0.333 0.333 0.5   0.417 0.194 0.361 0.389
#  0.389 0.528 0.222 0.389 0.556 0.417 0.778 0.556 0.611 0.917 0.167 0.833
#  0.667 0.806 0.611 0.583 0.694 0.389 0.417 0.583 0.611 0.944 0.944 0.472
#  0.722 0.361 0.944 0.556 0.667 0.806 0.528 0.5   0.583 0.806 0.861 1.
#  0.583 0.556 0.5   0.944 0.556 0.583 0.472 0.722 0.667 0.722 0.417 0.694
#  0.667 0.667 0.556 0.611 0.528 0.444]
```

#### 30、如何计算 softmax 分数？ ☆☆☆
```python
# Softmax的含义在于不再唯一的确定某一个最大值，
# 而是为每个输出分类的结果都赋予一个概率值，
# 表示属于每个类别的可能性。
# 与之对应的Hardmax，则表示唯一最大值
import numpy as np
file = '../../data/analysis_tool/iris.csv'
data = np.genfromtxt(file, delimiter=',', dtype='object')
arr = np.array([float(row[0]) for row in data])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(arr))

# 输出：
# [0.002 0.002 0.001 0.001 0.002 0.003 0.001 0.002 0.001 0.002 0.003 0.002
#  0.002 0.001 0.004 0.004 0.003 0.002 0.004 0.002 0.003 0.002 0.001 0.002
#  0.002 0.002 0.002 0.002 0.002 0.001 0.002 0.003 0.002 0.003 0.002 0.002
#  0.003 0.002 0.001 0.002 0.002 0.001 0.001 0.002 0.002 0.002 0.002 0.001
#  0.003 0.002 0.015 0.008 0.013 0.003 0.009 0.004 0.007 0.002 0.01  0.002
#  0.002 0.005 0.005 0.006 0.004 0.011 0.004 0.004 0.007 0.004 0.005 0.006
#  0.007 0.006 0.008 0.01  0.012 0.011 0.005 0.004 0.003 0.003 0.004 0.005
#  0.003 0.005 0.011 0.007 0.004 0.003 0.003 0.006 0.004 0.002 0.004 0.004
#  0.004 0.007 0.002 0.004 0.007 0.004 0.016 0.007 0.009 0.027 0.002 0.02
#  0.011 0.018 0.009 0.008 0.012 0.004 0.004 0.008 0.009 0.03  0.03  0.005
#  0.013 0.004 0.03  0.007 0.011 0.018 0.007 0.006 0.008 0.018 0.022 0.037
#  0.008 0.007 0.006 0.03  0.007 0.008 0.005 0.013 0.011 0.013 0.004 0.012
#  0.011 0.011 0.007 0.009 0.007 0.005]
```

#### 31、






