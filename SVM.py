import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#读取数据
data = pd.read_csv('./train.csv',header=0)
label = np.array(data['label'])

#原始数据的描述
data.describe()

#原始数据描点画图
time = []
for i in range(1,6001):
    time.append(i)
data.head(1).drop(['label','id'],axis = 1).values
data.iloc[1].drop(["label","id"])
for i in range(0,100):
    if data.iloc[i]["label"] == 2:
        plt.plot(time, data.iloc[i].drop(["label","id"]).values)
        plt.ylabel('Vibration measurement')
        plt.xlabel(data.iloc[i]["label"])
        plt.show()

#切片得到训练集
train = data.iloc[:,1:-1]

#快速傅里叶变换，时域变换到频域
train = abs(np.fft.fft(train)[:,:3001])/6000

#快速傅里叶变换之后的数据图像
d1 = np.array(data.iloc[2,1:-1]) # 修改iloc[2,1:-1]中的第一个数据可以更换数据
n = len(d1)
yy = np.fft.fft(d1)
yy_ = yy[range(0,int(n/2))]
plt.plot(abs(yy_)/n)
plt.show()


#把训练集随机划分0.3作为测试集
x_train,x_test,y_train,y_test = train_test_split(train,label,random_state=1,train_size=0.7)


#数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


#构建SVM模型
clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovr')

#训练模型
clf.fit(x_train,y_train)

#预测结果准确率
print(clf.score(x_train,y_train))
print("训练集准确率：",accuracy_score(y_train,clf.predict(x_train)))
print("测试集准确率：",accuracy_score(y_test,clf.predict(x_test)))


#打印预测结果
result = clf.predict(x_test)
result = pd.DataFrame(result)
print(result)