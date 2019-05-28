# -*- coding: utf-8 -*-
"""
Created on 19-5-5 下午7:30

@author: chinbing <x62881999@gmail.com>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KNN:
	def __init__(self, points, k=3):
		self.points, self.labels = points
		self.inputs = None
		self.res = None
		self.k = k
		self.m = len(points)
		self.n = 0
		self.dimension = len(self.points[0])
		self.dist = None
		self.sorted_dist = None
		self.acc = 0.

	def _cal_dist(self, inputs):
		"""
		对输入列表inputs与标记数据points计算距离矩阵self.dist，并对距离索引到self.sorted_dist
		:param inputs: 要分类的数据点列表，shape为[n, 4]，可传多个点，非单个
		:return:
		"""
		self.m = len(self.points)
		self.n = len(inputs)
		tiled_points = np.tile(self.points, (self.n, 1)).reshape([self.n, self.m, self.dimension])
		tiled_inputs = np.tile(inputs, (1, self.m)).reshape([self.n, self.m, self.dimension])

		mat = (tiled_points - tiled_inputs) ** 2
		mat = mat.reshape([self.n, self.m, 2, 2])  # 将4个特征分为两组（花瓣与花萼）进行距离计算再求和
		mat = np.sqrt(np.sum(mat, axis=3))
		dist = np.sum(mat, axis=2)

		self.dist = dist
		self.sorted_dist = np.argsort(dist)

	def predict(self, inputs):
		"""
		对输入列表inputs进行knn分类，同时将结果返回与绑定到self.res中
		:param inputs: 要分类的数据点列表，shape为[n, 4]
		:return:
		"""
		self.inputs = inputs
		self._cal_dist(inputs)

		predicted = []
		for i in range(self.n):
			class_count = {}
			for j in range(self.k):
				vote_label = self.labels[self.sorted_dist[i][j]]

				class_count[vote_label] = class_count.get(vote_label, 0) + 1
			predicted.append(max(class_count, key=class_count.get))  # max选出字典value最大值对应的key

		self.res = predicted
		return predicted

	def evaluate(self, inputs, y):
		"""
		评估分类准确度
		:param inputs:
		:param y: inputs对应的ground truth labels
		:return:
		"""
		self.predict(inputs)
		tp = 0  # true positive
		for i in range(len(y)):
			if y[i] == self.res[i]:
				tp += 1
		self.acc = tp / len(y)

	def show(self):
		"""
		画图函数，没啥好说的，值得一提的是4维数据不好可视化，但考虑到前两个特征都是关于花瓣的，后两个都是关于花萼的，
		所以我把4个特征分成两组分开绘制了两张图，其中圆点为训练数据，叉叉为测试数据，颜色为分类结果
		:return:
		"""
		# plt.rcParams[u'font.sans-serif'] = ['SimHei']
		# plt.rcParams['axes.unicode_minus'] = False
		fig = plt.figure('分类结果分别在花瓣-花萼上的可视化结果', figsize=(12, 5))
		fig.add_subplot(121)
		ax = fig.gca()
		ax.set_xlabel('x1')
		ax.set_ylabel('y1')

		setosa_x, setosa_y = [], []
		versicolor_x, versicolor_y = [], []
		virginica_x, virginica_y = [], []
		for i in range(len(self.labels)):
			if self.labels[i] == 'Iris-setosa':
				setosa_x.append(self.points[i][0])
				setosa_y.append(self.points[i][1])
			elif self.labels[i] == 'Iris-versicolor':
				versicolor_x.append(self.points[i][0])
				versicolor_y.append(self.points[i][1])
			elif self.labels[i] == 'Iris-virginica':
				virginica_x.append(self.points[i][0])
				virginica_y.append(self.points[i][1])

		ax.scatter(setosa_x, setosa_y, c='b', s=20, alpha=0.5)
		ax.scatter(versicolor_x, versicolor_y, c='r', s=20, alpha=0.5)
		ax.scatter(virginica_x, virginica_y, c='g', s=20, alpha=0.5)

		_setosa_x, _setosa_y = [], []
		_versicolor_x, _versicolor_y = [], []
		_virginica_x, _virginica_y = [], []
		for i in range(len(self.res)):
			if self.res[i] == 'Iris-setosa':
				_setosa_x.append(self.inputs[i][0])
				_setosa_y.append(self.inputs[i][1])
			elif self.res[i] == 'Iris-versicolor':
				_versicolor_x.append(self.inputs[i][0])
				_versicolor_y.append(self.inputs[i][1])
			elif self.res[i] == 'Iris-virginica':
				_virginica_x.append(self.inputs[i][0])
				_virginica_y.append(self.inputs[i][1])

		ax.scatter(_setosa_x, _setosa_y, marker='x', c='b', s=40, alpha=0.5)
		ax.scatter(_versicolor_x, _versicolor_y, marker='x', c='r', s=40, alpha=0.5)
		ax.scatter(_virginica_x, _virginica_y, marker='x', c='g', s=40, alpha=0.5)

		plt.legend(['setosa', 'versicolor', 'virginica'])

		plt.text(8.5, 1.6, 'accuracy:' + str(self.acc), family='serif', style='oblique', color='green', fontsize=12, ha='center')

		fig.add_subplot(122)
		ax = fig.gca()
		ax.set_xlabel('x2')
		ax.set_ylabel('y2')

		setosa_x, setosa_y = [], []
		versicolor_x, versicolor_y = [], []
		virginica_x, virginica_y = [], []
		for i in range(len(self.labels)):
			if self.labels[i] == 'Iris-setosa':
				setosa_x.append(self.points[i][2])
				setosa_y.append(self.points[i][3])
			elif self.labels[i] == 'Iris-versicolor':
				versicolor_x.append(self.points[i][2])
				versicolor_y.append(self.points[i][3])
			elif self.labels[i] == 'Iris-virginica':
				virginica_x.append(self.points[i][2])
				virginica_y.append(self.points[i][3])

		ax.scatter(setosa_x, setosa_y, c='b', s=20, alpha=0.5)
		ax.scatter(versicolor_x, versicolor_y, c='r', s=20, alpha=0.5)
		ax.scatter(virginica_x, virginica_y, c='g', s=20, alpha=0.5)

		_setosa_x, _setosa_y = [], []
		_versicolor_x, _versicolor_y = [], []
		_virginica_x, _virginica_y = [], []
		for i in range(len(self.res)):
			if self.res[i] == 'Iris-setosa':
				_setosa_x.append(self.inputs[i][2])
				_setosa_y.append(self.inputs[i][3])
			elif self.res[i] == 'Iris-versicolor':
				_versicolor_x.append(self.inputs[i][2])
				_versicolor_y.append(self.inputs[i][3])
			elif self.res[i] == 'Iris-virginica':
				_virginica_x.append(self.inputs[i][2])
				_virginica_y.append(self.inputs[i][3])

		ax.scatter(_setosa_x, _setosa_y, marker='x', c='b', s=40, alpha=0.5)
		ax.scatter(_versicolor_x, _versicolor_y, marker='x', c='r', s=40, alpha=0.5)
		ax.scatter(_virginica_x, _virginica_y, marker='x', c='g', s=40, alpha=0.5)

		plt.legend(['setosa', 'versicolor', 'virginica'])

		plt.show()


if __name__ == '__main__':
	iris = pd.read_csv('iris.data', header=None)
	labels = np.array(iris[4])  # label列
	data = np.array(iris.iloc[:, 0:4])  # 特征列

	# 均匀切片
	_points = np.concatenate([data[:33], data[67:-16]], axis=0)
	_labels = np.concatenate([labels[:33], labels[67:-16]], axis=0)

	_inputs = np.concatenate([data[33:67], data[-16:]], axis=0)
	_y = np.concatenate([labels[33:67], labels[-16:]], axis=0)

	knn = KNN([_points, _labels], k=2)
	knn.evaluate(_inputs, _y)
	print('acc: ', knn.acc)
	knn.show()

	input('press any key to exit...')

