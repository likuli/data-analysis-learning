在掌握Numpy基础之后，为了能将Numpy快速应用于实践，特意梳理了这100道练习，供大家学习。

练习题难易程度分为四个级别，其中☆最简单，☆☆☆☆最难。

记住：看，没用！码起来才是正事儿！

练习题难度分布：

| 难度 | 题量 | 百分比 |
| ---- | ------- | ------- |
|☆ | 0 | 25%|
|☆☆| 0 | 25%|
|☆☆☆| 0 ||
|☆☆☆☆ | 0 ||

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





