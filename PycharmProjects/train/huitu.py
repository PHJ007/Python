#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np


n=np.loadtxt('result.txt')#加载数据
print n
print(n.sum(axis=1))
m=n.sum(axis=1)#以行累加
n=-(n.T/m).T#将每一行平均
print n
fig,ax=plt.subplots()
cmap=mpl.cm.gray#以灰度显示
ax.imshow(n,cmap=cmap)
plt.show()


