import numpy as np

# demo 1: 打印numpy版本及配置
print(np.__version__)
np.show_config()

# demo 2: 创建长度为10的向量
z = np.zeros(10)
print(z)


# demo 3: 查看数组内存大小
z1 = np.zeros((10, 10), dtype=int)  # 10x10 int矩阵
z2 = np.zeros((10, 10), dtype=float)  # 10x10 float矩阵
print("z1 memory： %d bytes" % (z1.size * z1.itemsize))
print("z2 memory： %d bytes" % (z2.size * z2.itemsize))


# demo 4: 查看numpy函数使用文档
np.info(np.zeros)
np.info(np.add)


# demo 5: 创建长度为10的向量，并将第5个赋值为“1”
z = np.zeros(10, dtype=int)
z[4] = 1
print(z)


# demo 6: 创建值域范围为10~49的向量
z = np.arange(10, 50)
print(z)


# demo 7: 向量反转
z = np.arange(10)
print(z)
z = z[::-1]
print(z)


# demo 8: 创建3x3矩阵，值域0~8
z = np.arange(9).reshape(3,3)
print(z)


# demo 9: 找到数组非0元素位置索引
arr = [1, 2, 0, 6, 0, 7, 10, 0, 4, 2]
nz = np.nonzero(arr)
print(nz)


# demo 10: 创建3x3单位矩阵
z = np.eye(3)
print(z)


# demo 11: 创建一个 3x3x3的随机矩阵
z = np.random.random((3, 3, 3))
print(z)


# demo 12: 创建一个 10x10的随机整数(0~10000)矩阵，并取最大值和最小值
z = np.random.randint(0, 10000, size=(10, 10))
print(z)
zmin, zmax = z.min(), z.max()
print(zmin, zmax)


# demo 13: 创建长度20的随机数组，并计算平均值
z = np.random.random(20)
print(z)
m = z.mean()
print(m)


# demo 14: 创建二维数组，边界值为1，其余为0
z = np.ones((10, 10))
z[1:-1, 1:-1] = 0
print(z)


# demo 15: 对已有数组，添加用0填充的边界
z = np.ones((5, 5))
z = np.pad(z, pad_width=1, mode='constant', constant_values=0)
print(z)


# demo 16: 查看下面表达式运行结果
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
print(np.nan)


# demo 17: 创建5x5矩阵，设置1/2/3/4落在对角线下方位置
z = np.diag(1+np.arange(4), k=-1)
print(z)


# demo 18: 创建8x8矩阵，并设置成棋盘样式
z = np.zeros((8, 8), dtype=int)
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print(z)


# demo 19: 获取(6,7,8)形状的矩阵中，第100个元素在矩阵中的索引
# 第100个元素，表示讲原多维矩阵，看成一维矩阵的数值
z = np.unravel_index(100, (6, 7, 8))
print(z)


# demo 20: 用tile函数创建8x8棋盘样式矩阵
z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(z)


# demo 21: 5x5矩阵归一化
# 归一化：将矩阵的数据以列为单元，按照一定比例，映射到某一区间，以达到无量纲化的效果。
# 无量纲化：是指通过一个合适的变量替代，将一个涉及物理量的方程的部分或全部的单位移除，以求简化实验或者计算的目的，是科学研究中一种重要的处理思想。
z = np.random.random((5, 5))
print(z)
zmax, zmin = z.max(), z.min()
z = (z - zmin) / (zmax - zmin)
print(z)


# demo 22: 创建一个将颜色描述为(RGBA)四个无符号字节的自定义dtype
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
z = np.zeros((2, 2), dtype=color)
print(z)


# demo 23: 矩阵乘积
# 矩阵乘法竟然忘了，复习了一下
z1 = np.ones((5, 3))
z2 = np.ones((3, 2))
z = np.dot(z1, z2)
print(z)


# demo 24: 给定数组，对大于3但小于8（3＜x＜8）的元素取反
z = np.arange(11)
print(z)
z[(3 < z) & (z < 8)] *= -1
print(z)


# demo 25: 获取下述脚本运行结果
# -1表示：元组求和,然后减1
print(sum(range(5), -1))
# -1表示：当前数组最后一个维度求和，因为当前是一维数组，也就是第一维度求和
print(np.sum(range(5), -1))


# demo 26: 执行表达式，查看结果
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))


# demo 27: 四舍五入
z = np.random.uniform(-10, +10, 10)
print(z)
print(np.copysign(np.ceil(np.abs(z)), z))


# demo 28: 获取两个数组中的相同元素
z1 = np.random.randint(0, 10, 10)
z2 = np.random.randint(0, 10, 10)
print(z1, z2)
z = np.intersect1d(z1, z2)
print(z)


# demo 29: 获取昨天、今天、明天日期
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday)
print(today)
print(tomorrow)


# demo 30: 获取2016年7月的所有日期
# 2016-07-01 ≤ x ＜ 2016-08-01
z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(z)


# demo 31: 如何直接在位计算(A+B)*(-A/2)(不建立副本)?
A = np.ones(3) * 1
B = np.ones(3) * 2

np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
print(A)


# demo 32: 用五种不同的方法去提取一个随机数组的整数部分
z = np.random.uniform(0, 10, 10)
print(z)

# 1.
print(z - z % 1)

# 2.返回不大于输入参数的最大整数
print(np.floor(z))

# 3.对数组中的元素向上取整
print(np.ceil(z)-1)

# 4.数组数据类型转换
print(z.astype(int))

# 5.返回输入数组元素的截断值
print(np.trunc(z))


# demo 33: 创建一个5x5的矩阵，其中每行数值 取值范围从0到4
# 按顺序排列？
z = np.zeros((5, 5))
z += np.arange(5)
print(z)

# 随机取值
z = np.zeros((5, 5))
z += np.random.randint(0, 5, size=5)
print(z)


# demo 34: 通过可生成10个数字的函数，来构建一个数组
def generage():
    for x in range(10):
        yield x

z = np.fromiter(generage(), dtype=int, count=-1)
print(z)


# demo 35: 创建长度为10的随机向量，值域从0到1，不包括0和1
z = np.linspace(0, 1, 11, endpoint=False)[1:]
print(z)


# demo 36: 创建随机向量，并排序
z = np.random.random(10)
z.sort()
print(z)


# demo 37: 对于小数组，如何用比np.sum更快的方式对其求和？
z = np.arange(10)
print(z)
print(np.add.reduce(z))


# demo 38: 检查两个随机数组是否相等
a = np.random.randint(0, 2, 5)
b = np.random.randint(0, 2, 5)
print(a, b)
e = np.allclose(a, b)
print(e)


# demo 39: 创建一个只读数组
z = np.zeros(10)
z.flags.writeable = False
z[0] = 1  # ValueError: assignment destination is read-only


# demo 40:  将笛卡尔坐标下的一个10x2的矩阵转换为极坐标形式
# 复习了笛卡尔坐标和极坐标的概念
z = np.random.random((10, 2))
print(z)
x, y = z[:, 0], z[:, 1]
r = np.sqrt(x**2+y**2)
t = np.arctan2(y, x)
print(r)
print(t)


# demo 41: 创建一个长度为10的向量，并将向量中最大值替换为1
z = np.random.random(10)
print(z)
z[z.argmax()] = 1
print(z)


# demo 42: 创建一个结构化数组，并实现x和y坐标覆盖[0,1]x[0,1]区域
z = np.zeros((5, 5), [('x', float), ('y', float)])
print(z)
# meshgrid: 生成网格点坐标矩阵
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print(z)


# demo 43:  给定两个数组x和y，构造Cauchy矩阵
# Cauchy: 柯西矩阵
x = np.arange(8)
y = x + 0.5
print(x, y)
# subtract：矩阵减法运算
c = 1.0 / np.subtract.outer(x, y)
print(np.linalg.det(c))


# demo 44: 打印每个numpy标量类型的最小值和最大值
# iinfo()函数显示整数类型的机器限制
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# demo 45: 打印一个数组中的所有数值
# set_printoptions 控制打印输出格式
np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=6)
z = np.zeros((16, 16))
print(z)


# demo 46: 给定标量，找到数组中最接近标量的值
z = np.arange(100)
print(z)
v = np.random.uniform(0, 100)
print(v)
# argmin 在指定维度（轴）上的最小值对应的位置索引
index = (np.abs(z-v)).argmin()
print(index)
print(z[index])


# demo 47: 创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组
z = np.zeros(10, [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color', [('r', float, 1),
                            ('g', float, 1),
                            ('b', float, 1)])])
print(z)


# demo 48: 对一个表示坐标形状为(10,2)的随机向量，找到点与点的距离
# atleast_2d: 将输入视为至少具有二维的数组。
z = np.random.random((10, 2))
x, y = np.atleast_2d(z[:, 0], z[:, 1])
d = np.sqrt((x - x.T)**2 + (y - y.T)**2)
print(d)


# demo 49: 将32位的浮点数(float)转换为对应的整数(integer)
z = np.arange(10, dtype=np.float32)
z = z.astype(np.int32, copy=False)
print(z)


# demo 50: enumerate的等价操作
z = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(z):
    print(index, value)
for index in np.ndindex(z.shape):
    print(index, z[index])


# demo 51: 生成一个通用的二维Gaussian-like数组
# 开始理解困难了~~
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
d = np.sqrt(x*x + y*y)
sigma, mu = 1.0, 0.0
g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
print(g)


# demo 52: 在二维数组内部随机放置p个元素
n = 10
p = 3
z = np.zeros((n, n))
np.put(z, np.random.choice(range(n*n), p, replace=False), 88)
print(z)


# demo 53: 减去一个矩阵中 每一行的平均值
x = np.random.rand(5, 10)
# mean 求平均值
y = x - x.mean(axis=1, keepdims=True)
print(y)


# demo 54: 根据第n列对数组进行排序
z = np.random.randint(0, 10, (3, 3))
print(z)
# 根据第二列排序
print(z[z[:, 1].argsort()])


# demo 55: 检查二维数组是否有空列
# 全部为0 则为空列
z = np.random.randint(0, 3, (3, 10))
print((~z.any(axis=0)).any())


# demo 56: 从数组中的给定值中找出最近的值
z = np.random.uniform(0, 1, 10)
print(z)
v = 0.6
# flat 将数组转换为迭代器
m = z.flat[np.abs(z-v).argmin()]
print(m)


# demo 57: 用迭代器(iterator)计算两个分别具有形状(1,3)和(3,1)的数组
a = np.arange(3).reshape(1, 3)
b = np.arange(3).reshape(3, 1)
# nditer 数组迭代器
it = np.nditer([a, b, None])
for x, y, z in it:
    z[...] = x + y
print(it.operands[2])


# demo 58: 创建一个具有name属性的数组类
class NameArray(np.ndarray):
    def __new__(cls, array, name=None):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', None)


z = NameArray(np.arange(10), "Test")
print(z)
print(z.name)


# demo 59: 考虑一个给定的向量，如何对第二个向量索引的每个元素加1
# 题目没看懂，但是理解了bincount()的用法
# bincount 索引值在x中出现的次数
z = np.ones(10)
print(z)
i = np.random.randint(0, len(z), 20)
print(i)
z += np.bincount(i, minlength=len(z))
print(z)


# demo 60: 根据索引列表(I)，将向量(X)的元素累加到数组(F)
x = [1, 2, 3, 4, 5, 6]
i = [1, 4, 9, 2, 4, 1]
f = np.bincount(i, x)
print(f)


# demo 61: 考虑一个(dtype=ubyte) 的 (w,h,3)图像，计算其唯一颜色的数量
w, h = 16, 16
i = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
f = i[..., 0] * (256 * 256) + i[..., 1] * 256 + i[..., 2]
n = len(np.unique(f))
print(n)


# demo 62: 如何一次性计算四维数组最后两个轴(axis)的和
a = np.random.randint(0, 10, (3, 4, 3, 4))
print(a.sum(axis=(-2, -1)))
print(a.reshape(a.shape[:-2] + (-1,)).sum(axis=-1))


# demo 63: 一维向量D，如何使用相同大小的向量S来计算D子集的均值
d = np.random.uniform(0, 1, 100)
s = np.random.randint(0, 10, 100)
d_sums = np.bincount(s, weights=d)
d_counts = np.bincount(s)
d_means = d_sums / d_counts
print(d_sums)
print(d_counts)
print(d_means)


# demo 64: 如何获得点积 dot prodcut的对角线
a = np.random.randint(0, 10, (5, 5))
b = np.random.randint(0, 10, (5, 5))
c = np.dot(a, b)
print(np.diag(c))
d = np.sum(a * b.T, axis=1)
print(d)


# demo 65: 向量[1,2,3,4,5],建立新向量，在新向量中每个值之间有3个连续零
z = np.array([1, 2, 3, 4, 5])
nz = 3
z0 = np.zeros(len(z) + (len(z) - 1) * nz)
z0[::nz+1] = z
print(z0)


# demo 66: 如何将维度(5,5,3)的数组与一个(5,5)的数组相乘
a = np.ones((5, 5, 3))
b = 2 * np.ones((5, 5))
print(a * b[:, :, None])


# demo 67: 交换数组中任意两行的位置
a = np.arange(25).reshape(5, 5)
a[[0, 2]] = a[[2, 0]]
print(a)


# demo 68: 找到数组中出现频率最高的值
z = np.random.randint(0, 10, 50)
print(np.bincount(z).argmax())


# demo 69: 从10x10的矩阵中提取出连续3x3区块
z = np.random.randint(0, 5, (10, 10))
n = 3
i = 1 + (z.shape[0] - 3)
j = 1 + (z.shape[1] - 3)
c = np.lib.stride_tricks.as_strided(z, shape=(i, j, n, n), strides=z.strides + z.strides)
print(c)


# demo 70: 找到数组第n个最大值
z = np.arange(10000)
np.random.shuffle(z)
n = 5
print(z[np.argsort(z)[-n:]])


# demo 71: 找到数组每列最大值
z = np.random.randint(0, 100, (3, 3))
print(z)
r = np.amax(z, axis=0)
print(r)


# demo 72: 找到数组每行最小值
z = np.random.randint(0, 100, (5, 5))
print(z)
r = np.amin(z, axis=1)
print(r)


# demo 73: 提取数组中每个元素出现的次数
z = np.random.randint(0, 10, 10)
print(z)
r = np.unique(z, return_counts=True)
print(r)


# demo 74: 将数组按行重复一次
z = np.random.randint(0, 10, (3, 3))
r = np.repeat(z, 2, axis=0)
print(r)


# demo 75: 去重数组重复行
z = np.random.randint(0, 10, (3, 3))
r = np.repeat(z, 2, axis=0)
print(r)
t = np.unique(r, axis=0)
print(t)


# demo 76: 从数组第一行中不放回抽3个元素
z = np.random.randint(0, 10, (5, 5))
print(z)
r = np.random.choice(z[0:1][0], 3, replace=False)
print(r)


# demo 77: 提取数组第二行中不含第三行元素的元素
z = np.random.randint(0, 10, (5, 5))
print(z)
a = z[1:2]
b = z[2:3]
print(a, b)
ind = np.isin(a, b)
arr = a[~ind]
print(arr)


# demo 78: 将数组每行升序排列
z = np.random.randint(0, 10, (5, 5))
print(z)
z.sort(axis=1)
print(z)


# demo 79: 将小于5的元素修改为nan
z = np.random.uniform(0, 10, (5, 5))
print(z)
z[z < 5] = np.nan
print(z)


# demo 80: 删除数组中含有NaN的行
z = np.random.uniform(0, 10, (5, 5))
z[z < 1] = np.nan
print(z)
z = z[~np.isnan(z).any(axis=1), :]
print(z)


# demo 81: 找出第一行出现频率最高的值
z = np.random.randint(0, 10, (10, 10))
val, count = np.unique(z[0, ::], return_counts=True)
print(val, count)
print(val[np.argmax(count)])


# demo 82: 找到数组中与100最接近的数字
z = np.random.randint(0, 100, (5, 5))
print(z)
a = z.flat[np.abs(z-100).argmin()]
print(a)


# demo 83: 数组每行元素减去该行平均值
z = np.random.randint(0, 100, (5, 5))
a = z - z.mean(axis=1, keepdims=True)
print(a)


# demo 84: 讲数组存储至本地文件
z = np.random.randint(0, 10, (3, 3))
np.savetxt('demo.txt', z)


# demo 85: 使用numpy进行描述性统计分析
# 相关矩阵也叫相关系数矩阵，其是由矩阵各列间的相关系数构成
# 相关系数，研究变量之间线性相关程度的量
# 协方差矩阵的每个元素是各个向量元素之间的协方差
# 协方差在概率论和统计学中用于衡量两个变量的总体误差
arr1 = np.random.randint(0, 10, 5)
arr2 = np.random.randint(0, 10, 5)
print(arr1, arr2)
print("arr1的平均数：%s" % np.mean(arr1))
print("arr1的中位数：%s" % np.median(arr1))
print("arr1的方差：%s" % np.var(arr1))
print("arr1的标准差：%s" % np.std(arr1))
print("arr1,arr2的相关矩阵：%s" % np.cov(arr1, arr2))
print("arr1,arr2的协方差矩阵：%s" % np.corrcoef(arr1, arr2))


# demo 86: 创建副本
z = np.array([1, 2, 3, 4, 5])
z_copy = z.copy()
print(z_copy)


# demo 87: 数组切片
arr = np.arange(10)
print(arr)
# slice(start, stop[, step])
a = slice(2, 8, 2)
r = arr[a]
print(r)


# demo 88: Numpy操作字符串,字符串拼接,首字母大写
str1 = ['I love']
str2 = [' python']
str3 = np.char.add(str1, str2)
print(str3)

str4 = np.char.title(str3)
print(str4)


# demo 89: 分别对二维数组的行、列逆序
a = np.random.randint(0, 10, (4, 4))
print(a)
# 列逆序
print(a[:, -1::-1])
# 行逆序
print(a[-1::-1, :])


# demo 90: 多条件筛选数据
a = np.random.randint(0, 20, 10)
print(a)
print(a[(a > 5) & (a < 15)])


# demo 91: 数据向上/下取整
a = np.random.uniform(0, 10, 10)
# 向上取整
print(np.ceil(a))
# 向下取整
print(np.floor(a))


# demo 92: Numpy概率抽样
a = np.array([1, 2, 3, 4, 5])
r = np.random.choice(a, 20, p=[0.1, 0.1, 0.1, 0.1, 0.6])
print(r)


# demo 93: 创建等差数列
a = np.linspace(start=5, stop=50, num=10)
print(a)


# demo 94: numpy数组分割
a = np.arange(12).reshape((3, 4))
print(a)
# 等量分割
print(np.split(a, 2, axis=1))
# 不等量分割
print(np.array_split(a, 3, axis=1))


# demo 95: Numpy压缩矩阵
arr = np.random.randint(1, 10, [3, 1])
print(arr)
print(np.squeeze(arr))


# demo 96: 使用numpy求解线性方程组
a = np.array([[1, 2, 3], [2, -1, 1], [3, 0, -1]])
b = np.array([9, 8, 3])
x = np.linalg.solve(a, b)
print(x)


# demo 97: 使用NumPy对数组分类,将大于等于7，或小于3的元素标记为1，其余为0
arr = np.random.randint(1,20,10)
print(arr)
print(np.piecewise(arr, [arr < 3, arr >= 7], [-1, 1]))


# demo 98: 数组标准化处理
a = np.random.randint(0, 10, (3, 3))
mu = np.mean(a, axis=0)
sigma = np.std(a, axis=0)
r = (a - mu) / sigma
print(r)


# demo 99: 归一化至区间[0,1]
n = np.random.randint(0, 10, (3, 3))
a = np.max(n) - np.min(n)
r = (n - np.min(n)) / a
print(r)




