from __future__ import print_function
from sklearn import svm
import numpy as np
import csv
import matplotlib.pyplot as plt
from svm_model import *

####################### Part 1 - read and plot data ##########################
print('Loading and Visualizing Data ...\n')
with open('D:\code\python\svm_python\ex1\ex1data1.csv') as csvfile:
    read_file1 = csv.reader(csvfile, delimiter=',')
    X = []
    y = []
    data1 = []
    data2 = []
    for row in read_file1:
        value1 = [float(row[0]), float(row[1])]
        value2 = row[2]
        X.append(value1)
        y.append(value2)
        if int(row[2]) == 1:
            data1.append([float(row[0]), float(row[1])])
        else:
            data2.append([float(row[0]), float(row[1])])
x1 = []
x2 = []
y1 = []
y2 = []
for i in range(len(data1)):
    x1.append(data1[i][0])
for i in range(len(data1)):
    y1.append(data1[i][1])
for i in range(len(data2)):
    x2.append(data2[i][0])
for i in range(len(data2)):
    y2.append(data2[i][1])
plt.scatter(x1, y1, c='red')
plt.scatter(x2, y2, c='blue')
plt.show()
input('Program paused. Press enter to continue.\n')

######################## Part 2 - training linear SVM #########################
print('\nTraining Linear SVM ...\n')
X = np.asarray(X)
y = np.asarray(y)
clf = svm_linear(X, y)
clf.fit(X, y)

w = clf.coef()
a = -w[0] / w[1]
xx = np.linspace(0, 6)
yy = a * xx - (clf.bias() / w[1])

plt.scatter(x1, y1, c='red')
plt.scatter(x2, y2, c='blue')
plt.plot(xx, yy, 'k-')
plt.show()
input('Program paused. Press enter to continue.\n')

######################## Part 3: Implementing Gaussian Kernel #########################
