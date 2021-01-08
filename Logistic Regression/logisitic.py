import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import *
# 定义sigmoid函数
def sigmoid(n):
    return 1.0/(1+exp(-n))

def getdata():
    # 读取数据集 并且把数据集分为测试集和训练集，奇数项为测试集，偶数项为训练集
    dataSet = []
    labels = []
    f = open('Iris.csv')
    flag = 1  # 设置flag来跳过第一行
    for line in f.readlines():
        if(flag == 1):
            flag = 0
            continue
        str_data = line.strip().split(",")[1:]
        labels.append(str_data[-1])
        for k in range(len(str_data)-1):
            str_data[k] = float(str_data[k])
        str_data.insert(0, 1.0)
        dataSet.append(str_data)
    train_data = []
    test_data = []
    train_label = []
    dataset = dataSet
    # 归一化处理优点：1.加快求解速度 2.可能提高精度
    for j in range(1, len(dataset[0])-1):
        l_sum = 0
        all_sum = 0
        for i in range(len(dataset)):
            l_sum += dataset[i][j]
        # 计算期望
        average = l_sum/len(dataset)
        # 计算标准差
        for i in range(len(dataset)):
            all_sum += pow(dataset[i][j]-average, 2)
        standard_deviation = math.sqrt(all_sum/(len(dataset)-1))
        # 对每列数据进行归一化,减均值，除方差
        for i in range(len(dataset)):
            dataset[i][j] = (dataset[i][j]-average)/standard_deviation
    for i in range(len(dataSet)):
        if i % 2 == 0:
            train_data.append(dataSet[i])
            train_label.append(labels[i])
        else:
            test_data.append(dataSet[i])
       #返回参数中，dataSet为全部数据，便于西瓜数据集画图，train_data为训练集，test_data为数据集
    return dataSet, labels,  train_data, test_data, train_label


def get_mostfeature(lt):
    index1 = 0  # 记录出现次数最多的元素下标
    max = 0  # 记录最大的元素出现次数
    for i in range(len(lt)):
        flag = 0  # 记录每一个元素出现的次数
        for j in range(i+1, len(lt)):  # 遍历i之后的元素下标
            if lt[j] == lt[i]:
                flag += 1  # 每当发现与自己相同的元素，flag+1
        if flag > max:  # 如果此时元素出现的次数大于最大值，记录此时元素的下标
            max = flag
            index1 = i
    return lt[index1]


def get_weight(data, label):
    matrix_data = mat(data)  # m行n列
    matrix_label = mat(label).transpose()  # m行
    m, n = shape(matrix_data)
    learn_rate = 0.00001  # 学习率
    max_cycles = 60000  # 迭代次数
    weights = ones((n, 1))  # 初始化权值矩阵
    # 利用梯度下降法进行权重更新
    for k in range(max_cycles):
        estimate_result = sigmoid(matrix_data*weights)
        error = matrix_label-estimate_result
        weights = weights+learn_rate*matrix_data.transpose()*error
    return weights


# 训练分类器
def train_model(data, labels):
    # 最终的训练结果形式：[[分类器1],[分类器2],[分类器3]...]
    # 把第i类作为正类，其他作为负类
    result_weight = []
    uniqueVals = list(set(labels))
    for i in range(len(uniqueVals)):
        for k in range(i+1, len(uniqueVals)):
            train_data = []
            train_label = []
            for j in range(len(data)):
                if(data[j][-1] == uniqueVals[i]):
                    train_label.append(1.0)
                    train_data.append(data[j][:-1])
                elif(data[j][-1] == uniqueVals[k]):
                    train_label.append(0.0)
                    train_data.append(data[j][:-1])
            re = get_weight(train_data, train_label).tolist()

            re.append(uniqueVals[i])
            re.append(uniqueVals[k])

            result_weight.append(re)

    return result_weight


def predict(pre_data, result_weight, label):
    pre_data = mat(pre_data)
    uniqueVals = list(set(label))
    result = []
    for i in range(len(result_weight)):
        estimate_result = sigmoid(pre_data*mat(result_weight[i][:-2]))
        if(estimate_result >= 0.5):
            # 正例
            result.append(result_weight[i][-2])
        else:
            # 反例
            result.append(result_weight[i][-1])
    return get_mostfeature(result)


def test_rate(test_data):
    postive = 0
    total = len(test_data)
    for i in range(total):
        pre_data = test_data[i][:-1]
        result = predict(pre_data, result_weight, train_label)
        if(result == test_data[i][-1]):
            postive = postive+1
    return postive/total

print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 获取训练集
all_data, all_labels, train_data, test_data, train_label = getdata()
print("训练中...")
# 训练模型
result_weight = train_model(train_data, train_label)
print("训练成功！训练结果为：", result_weight)
print("正在测试准确率...")
# 预测
rate = test_rate(test_data)
print("准确率:", rate)
ticks2 = time.time()
print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


'''
#画图，只有西瓜数据集（二维）才能画图，大于二维的数据集不能画图
weights = train_model(all_data, all_labels)
data_arr=array(all_data)
p=shape(data_arr)[0]
#print(p)
x1=[]
y1=[]
x2=[]
y2=[]
for i in range(p):
    if int(all_labels[i])==1:
        x1.append(float(data_arr[i,1]))
        y1.append(float(data_arr[i,2]))
    else:
        x2.append(float(data_arr[i,1]))
        y2.append(float(data_arr[i,2]))        
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(x1,y1,s=30,c='red',marker='s')
ax.scatter(x2,y2,s=30,c='green')
x=arange(-1.5,1.5,0.1)
# print(x)
y=(-weights[0][0][0]-weights[0][1][0]*x)/weights[0][2][0]
ax.plot(x,y)
plt.show()
'''
