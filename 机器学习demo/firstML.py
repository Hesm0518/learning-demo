import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 初步了解pandas

# 初步的了解数据
data = pd.read_csv('iris.data', header=None)
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
print(data.head())
print(data.describe())  # 已经进行初步的统计
print(data.info())  # 元数据

# 处理数据
data['label'] = 0
data.loc[data['class'] == 'Iris-setosa', 'label'] = 0
data.loc[data['class'] == 'Iris-versicolor', 'label'] = 1
data.loc[data['class'] == 'Iris-virginica', 'label'] = 2
data = data.drop('class', axis=1)
print(data['label'].unique())

X_train, X_test, Y_train, Y_test = train_test_split(data[['sepal length', 'sepal width', 'petal length', 'petal width']],
                                                    data['label'], test_size=0.2, shuffle=True)
# print(X_train, Y_train)

# 分类任务  kNN 朴素贝叶斯 神经网络 随机森林
nbrs = KNeighborsClassifier(n_neighbors=9)  # 模型

nbrs.fit(X_train, Y_train)
y_pred = nbrs.predict(X_test)

print(accuracy_score(Y_test, y_pred))

# 聚类任务 kMean


# 降维任务 PCA
