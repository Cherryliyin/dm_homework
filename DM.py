 # -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 20:22:51 2017

@author: xinxin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import operator

# **step1: 数据处理**
#
# 将原始的data文件转化为csv文件
# 从原始数据文件中读出，然后写入csv文件中

origin_fp= open("../data_set/horse-colic.data",'r')
modified_fp = open("../data_set/horse-colic.csv",'w')

line = origin_fp.readline()
while(line):
    temp = line.strip().split()
    temp = ','.join(temp) + '\n'
    modified_fp.write(temp)
    line = origin_fp.readline()
    
origin_fp.close()
modified_fp.close()


# 为每列数据添加别名
att_title = ["surgery","Age","HN","rt","pulse","rr","toe",
                 "pp","mm","crt","pain","peristalsis","ad","nt",
                 "nr","nrP","re","abdomen","pcv","tp","aa",
                 "atp","outcome","sl","ls1","ls2","ls3","cd"]
att_category = ["surgery","Age","toe","pp","mm","crt","pain","peristalsis",
                "ad","nt","nr","re","abdomen","aa","outcome","sl","cd"]
att_value = ["HN","rt","pulse","rr","nrP","pcv","tp","atp","ls1","ls2","ls3"]

att_value_num = ["rt","pulse","rr","nrP","pcv","tp","atp"]
att_value_str = ["HN","ls1","ls2","ls3"]

origin_data = pd.read_csv("../data_set/horse-colic.csv",
                          names = att_title,
                          index_col = False)
origin_data = origin_data.replace('?', np.nan)

# 将字符数据转换为category，以便进行统计
for item in att_category:
    origin_data[item] = origin_data[item].astype('category')

# print(origin_data)


# **step2: 数据摘要**
#
# ==> 对标称属性，给出每个可能取值的频数
for item in att_category:
    print(str(item) + '的频度为：\n' + str(pd.value_counts(origin_data[item].values)) + '\n')


# 数值属性
#
# 以下操作只针对数值型属性，不针对category型
#
# ==> 给出各属性最大值的个数：

for item in att_value:
    maxItem = origin_data[item].astype(float).max(skipna = True)
    maxNum = np.sum(origin_data[item].astype(float) == maxItem)
    print(str(item) + '的最大值为：' + str(maxItem) + '， 个数为：' + str(maxNum))

print("=================================")
    
# ==> 给出各属性最小值的个数：

for item in att_value:
    mixItem = origin_data[item].astype(float).min(skipna = True)
    minNum = np.sum(origin_data[item].astype(float) == mixItem)
    print(str(item) + '的最小值为：' + str(mixItem) + '， 个数为：' + str(minNum))
    
print("=================================")  

# ==> 给出各属性均值的个数：
 
for item in att_value:
    meanItem = origin_data[item].astype(float).mean(skipna = True)
    meanNum =  np.sum(origin_data[item].astype(float) == meanItem)
    print(str(item) + '的均值为：' + str(meanItem) + '， 个数为：' + str(meanNum))

print("=================================")   
    
# ==> 给出各属性中位数的个数：
   
for item in att_value:
    medianItem = origin_data[item].astype(float).median(skipna = True)
    medianNum =  np.sum(origin_data[item].astype(float) == medianItem)
    print(str(item) + '的中位数为：' + str(medianItem) + '， 个数为：' + str(medianNum))

print("=================================")   

# ==> 给出各属性四分位数的个数：

for item in att_value:    
    quartItem = origin_data[item][origin_data[item].isnull() == False].astype(float).describe().loc['25%']
    quartNum =  np.sum(origin_data[item].astype(float) == quartItem)
    print(str(item) + '的四分位数为：' + str(quartItem) + '， 个数为：' + str(quartNum))

print("=================================")  

# ==> 给出各属性缺失值的个数：
for item in att_value:    
    lostNum =  np.sum(origin_data[item].isnull())
    print(str(item) + '缺失值的个数为：'+ str(lostNum))


# **step3: 数据的可视化**
#
# 针对数值型属性，
# 绘制直方图，如mxPH，用qq图检验其分布是否为正太分布。
#
# 直方图

fig = plt.figure(figsize = (20,20))
i = 1
for item in att_value_num:
    ax = fig.add_subplot(3,3,i)
    origin_data[item].astype(float).plot(kind = 'hist', title = item, ax = ax)
    i += 1
fig.savefig("../data_set/hist.png")


# qq图

fig = plt.figure(figsize = (20,20))
i = 1
for item in att_value_num:
    ax = fig.add_subplot(3,3,i)
    sm.qqplot(origin_data[item].astype(float), ax = ax)
    ax.set_title(item)
    i += 1
fig.savefig("../data_set/qq.png")

## qq图满足正太分布的条件：
### qq图上的点近似地在一条直线附近，而且该直线的斜率为标准差，截距为均值
### 由此判断属性rt(rectal temperature) 和属性 pcv(packed cell volume)符合正态分布，其余均不符合
    

# 绘制盒图，对离群值进行识别

fig = plt.figure(figsize = (20,20))
i = 1
for item in att_value_num:
    ax = fig.add_subplot(3,3,i)
    origin_data[item].astype(float).plot(kind = 'box', title = item, ax = ax)
    i += 1
fig.savefig("../data_set/box.png")


# **Step 4. 数据缺失的处理**
#
### 数据集中有30%的值是缺失的，先处理数据中的缺失值
### 分别使用下列四种策略对缺失值进行处理:
#***** 1. 将缺失部分剔除
#***** 2. 用最高频率值来填补缺失值
#***** 3. 通过属性的相关关系来填补缺失值
#***** 4. 通过数据对象之间的相似性来填补缺失值

# 1. 将缺失部分剔除

# 对原始数据集进行拷贝
filtrated_data_1 = origin_data.copy()

## -使用dropna()整行删除缺失值
filtrated_data_1 = filtrated_data_1.dropna()

## -绘制可视化图像


## --对标称属性，绘制折线图

fig = plt.figure(figsize = (50,30))
i = 1
for item in att_category:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    pd.value_counts(origin_data[item].values).plot(ax = ax, marker = 'o', label = 'origin', legend = True)
    pd.value_counts(filtrated_data_1[item].values).plot(ax = ax, marker = '*', label = 'filtrated', legend = True)
    i += 1

    
## --对数值型属性，绘制直方图
i = 18
for item in att_value_num:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    origin_data[item].astype(float).plot(kind = 'hist', ax = ax, label = 'origin', legend = True, alpha = 0.5)
    filtrated_data_1[item].astype(float).plot(kind = 'hist', ax = ax, label = 'filtrated', legend = True, alpha = 0.5)
    ax.axvline(origin_data[item].astype(float).mean(), color = 'r')
    ax.axvline(filtrated_data_1[item].astype(float).mean(), color = 'b')
    i += 1

fig.savefig("../data_set/loss1.png")       
filtrated_data_1.to_csv('../data_set/filtrated_data_1.csv', mode = 'w', encoding = 'utf-8', index = False, header = False)


# 2. 用最高频率值来填补缺失值

# 对原始数据集进行拷贝
filtrated_data_2 = origin_data.copy()

# 对每一列，分别计算高频属性值
for item in att_title:
    most_frq_value = pd.value_counts(origin_data[item].values).idxmax()
    
    ## -使用fillna()替换缺失值
    filtrated_data_2[item].fillna(value = most_frq_value , inplace = True )
    
    
## -绘制可视化图像

## --对标称属性，绘制折线图

fig = plt.figure(figsize = (50,30))
i = 1
for item in att_category:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    pd.value_counts(origin_data[item].values).plot(ax = ax, marker = 'o', label = 'origin', legend = True)
    pd.value_counts(filtrated_data_2[item].values).plot(ax = ax, marker = '*', label = 'filtrated', legend = True)
    i += 1

    
## --对数值型属性，绘制直方图
i = 18
for item in att_value_num:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    origin_data[item].astype(float).plot(kind = 'hist', ax = ax, label = 'origin', legend = True, alpha = 0.5)
    filtrated_data_2[item].astype(float).plot(kind = 'hist', ax = ax, label = 'filtrated', legend = True, alpha = 0.5)
    ax.axvline(origin_data[item].astype(float).mean(), color = 'r')
    ax.axvline(filtrated_data_2[item].astype(float).mean(), color = 'b')
    i += 1
    
fig.savefig("../data_set/loss2.png")       
filtrated_data_2.to_csv('../data_set/filtrated_data_2.csv', mode = 'w', encoding = 'utf-8', index = False, header = False)


# 3. 通过属性的相关关系来填补缺失值

# 对原始数据集进行拷贝
filtrated_data_3 = origin_data.copy()

# 使用 interpolate()，对数值属性进行插值法替换缺失值
for item in att_value_num:
    filtrated_data_3[item].interpolate(inplace = True)

## -绘制可视化图像

## --对标称属性，绘制折线图

fig = plt.figure(figsize = (50,30))
i = 1
for item in att_category:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    pd.value_counts(origin_data[item].values).plot(ax = ax, marker = 'o', label = 'origin', legend = True)
    pd.value_counts(filtrated_data_3[item].values).plot(ax = ax, marker = '*', label = 'filtrated', legend = True)
    i += 1

    
## --对数值型属性，绘制直方图
i = 18
for item in att_value_num:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    origin_data[item].astype(float).plot(kind = 'hist', ax = ax, label = 'origin', legend = True, alpha = 0.5)
    filtrated_data_3[item].astype(float).plot(kind = 'hist', ax = ax, label = 'filtrated', legend = True, alpha = 0.5)
    ax.axvline(origin_data[item].astype(float).mean(), color = 'r')
    ax.axvline(filtrated_data_3[item].astype(float).mean(), color = 'b')
    i += 1
    
fig.savefig("../data_set/loss3.png")       
filtrated_data_3.to_csv('../data_set/filtrated_data_3.csv', mode = 'w', encoding = 'utf-8', index = False, header = False)


"""
# 4. 通过数据对象之间的相似性来填补缺失值

# 对原始数据集进行拷贝，用来进行正则化处理
copy_data = origin_data.copy()

# 将备份数据集中数值属性的缺失值替换为0
copy_data[att_value_num] = copy_data[att_value_num].fillna(0)

# 对数据进行正则化：
copy_data[att_value_num] = copy_data[att_value_num].astype(float).apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))



# 构造分数表
score = {}
range_length = len(copy_data)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0


# 在处理后的数据中，计算两条数据的差异性得分，分值越高，差异性越大
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in att_category:
            if copy_data.iloc[i][item] != copy_data.iloc[j][item]:
                score[i][j] += 1
        for item in att_value_num:
            temp = abs((copy_data.iloc[i][item]).astype(float) - (copy_data.iloc[j][item]).astype(float))
            score[i][j] += temp
        score[j][i] = score[i][j]



# 建立原始数据集的拷贝
filtrated_data_4 = origin_data.copy()

# 查找所有具有缺失值的条目
list_nan = pd.isnull(origin_data).any(1).nonzero()[0]

# 对有缺失值的条目，用与之相似度最高（得分最低）的数据条目对应的属性值进行替换
for index in list_nan:
    similar = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
    for item in att_value:
        if pd.isnull(filtrated_data_4.iloc[index][item]):
            if pd.isnull(origin_data.iloc[similar][item]):
                filtrated_data_4.ix[index,item] = origin_data[item].value_counts().idmax()
            else:
                filtrated_data_4.ix[index,item] = origin_data.iloc[similar][item]

## -绘制可视化图像

## --对标称属性，绘制折线图

fig = plt.figure(figsize = (50,30))
i = 1
for item in att_category:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    pd.value_counts(origin_data[item].values).plot(ax = ax, marker = 'o', label = 'origin', legend = True)
    pd.value_counts(filtrated_data_4[item].values).plot(ax = ax, marker = '*', label = 'filtrated', legend = True)
    i += 1

    
## --对数值型属性，绘制直方图
i = 18
for item in att_value_num:
    ax = fig.add_subplot(6,4,i)
    ax.set_title(item)
    origin_data[item].astype(float).plot(kind = 'hist', ax = ax, label = 'origin', legend = True, alpha = 0.5)
    filtrated_data_4[item].astype(float).plot(kind = 'hist', ax = ax, label = 'filtrated', legend = True, alpha = 0.5)
    ax.axvline(origin_data[item].astype(float).mean(), color = 'r')
    ax.axvline(filtrated_data_4[item].astype(float).mean(), color = 'b')
    i += 1
    
fig.savefig("../data_set/loss4.png")       
filtrated_data_4.to_csv('../data_set/filtrated_data_4.csv', mode = 'w', encoding = 'utf-8', index = False, header = False)
    
"""





