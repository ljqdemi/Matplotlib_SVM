import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# 生成线性可分的数据
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)

# 创建 SVM 分类器
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 创建一个网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

# 预测每个点的类
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和样本点
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
plt.title('SVM Decision Boundary', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.grid()
plt.show()