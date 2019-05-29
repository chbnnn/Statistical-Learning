# -*- coding: utf-8 -*-
"""
Created on 19-5-25 下午4:03

@author: chinbing <x62881999@gmail.com>
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from plot_tree import *

class ID3:
	def __init__(self):
		self.labels = []
		self.entropy = -1
		self.tree = None

	@staticmethod
	def calculate_entropy(data):
		label_array = data[:, -1]
		label_unique, counts = np.unique(label_array, return_counts=True)
		# label_count = dict(zip(label_unique, counts))
		m = len(data)
		entropy = 0.0
		for count in counts:
			prob_k = count / m
			info_k = -np.log(prob_k) / np.log(2)
			entropy += prob_k * info_k
		return entropy

	def calculate_conditional_entropy(self, data, column):
		unique_values = set(data[:, column])
		unique_indexs = [data[:, column] == value for value in unique_values]
		entropys = [sum(idxs) / len(data) * self.calculate_entropy(data[idxs]) for idxs in unique_indexs]
		conditional_entropy = sum(entropys)
		# print(self.labels[column], 'cond entropy is: ', conditional_entropy)
		return conditional_entropy

	# @staticmethod
	# def split_data_by_feature(data_origin, axis, value):
	# 	idx_vec = data_origin[:, axis] == value
	# 	data_reduced = np.delete(data_origin[idx_vec, :], axis, axis=1)
	# 	return data_reduced

	def find_best_feature(self, data, labels):
		base_entropy = self.calculate_entropy(data)
		max_info_gain = 0.0
		best_feature_index = None
		for i in labels[:-1]:
			# feature_value_list = data[:, i]
			# unique_feature_values = set(feature_value_list)
			# left_entropy = 0.0
			# for value in unique_values:
			# 	idxs = data[:, column] == value
			# 	prob_di = sum(idxs) / len(data)
			# 	info_di = self.calculate_entropy(data[idxs])
			# 	left_entropy += prob_di * info_di
			left_entropy = self.calculate_conditional_entropy(data, i)
			info_gain = base_entropy - left_entropy
			# print(self.labels[i], ' info gain: ', info_gain)
			if info_gain >= max_info_gain:  # >= rather than >
				max_info_gain = info_gain
				best_feature_index = i
		return best_feature_index

	@staticmethod
	def find_majority_class(class_list):
		label_unique, counts = np.unique(class_list, return_counts=True)
		label_count = dict(zip(label_unique, counts))
		majority_class = max(label_count, key=label_count.get)
		return majority_class

	def build_decision_tree(self, data, labels):
		class_list = data[:, -1]
		if len(set(class_list)) == 1:
			return class_list[0]
		# elif data.shape[1] == 1:
		elif len(labels) == 1:
			return self.find_majority_class(class_list)

		best_feature_index = self.find_best_feature(data, labels)
		best_feature_label = self.labels[best_feature_index]

		labels.remove(best_feature_index)
		tree = {best_feature_label: {}}

		feature_value_list = data[:, best_feature_index]
		unique_feature_value = np.unique(feature_value_list)
		for value in unique_feature_value:
			idxs = feature_value_list == value
			# if sum(idxs) == 0:
			# 	continue
			splited_data = data[idxs, :]
			tree[best_feature_label][value] = self.build_decision_tree(splited_data, labels)
		self.tree = tree
		return tree

	def fit(self, x, y):
		self.labels = y
		self.entropy = self.calculate_entropy(data=x)
		return self.build_decision_tree(data=x, labels=list(range(len(y))))

	def classify(self, test_vec, tree=None):
		if tree is None:
			tree = self.tree

		feature_root = list(tree.keys())[0]
		children = tree[feature_root]
		feature_index = self.labels.index(feature_root)

		for key in children.keys():
			if test_vec[feature_index] == key:
				if isinstance(children[key], dict):
					class_label = self.classify(test_vec, tree=children[key])
				else:
					class_label = children[key]
		return class_label

	def evaluate(self, x, y):
		y_pred = [self.classify(sample) for sample in x]
		acc = accuracy_score(y, y_pred)
		return acc



def createDataSet():
	"""
	创建一个简单的数据集。这个数据集根据两个属性来判断一个海洋生物是否
	属于鱼类, 第一个属性是不浮出水面是否可以生存, 第二个属性是是否有鳍。数据集
	中的第三列是分类结果。
	:return:
	"""
	labels = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
	data = np.array([['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否']])
	return data, labels


# 编写函数计算熵。计算公式看前面。
def calc_entropy(data):
	# 获取总的训练数据数
	m = len(data)
	# 创建一个字典统计各个类别的数据量
	label_count = {}
	for vec in data:
		# 使用下标-1 获取所属分类保存到 currentLabel
		current_label = vec[-1]
		# 若获得的类别属于新类别,则初始化该类的数据条数为 0
		if current_label not in label_count.keys():
			label_count[current_label] = 1
		else:
			label_count[current_label] += 1

	entropy = 0.0
	for key in label_count.keys():
		# 计算 p(xi)
		count_k = label_count[key]
		prob_k = count_k / m
		info_k = -np.log(prob_k) / np.log(2)
		# 计算熵
		# print(count_k, ' ', prob_k, ' ', info_k)
		entropy += prob_k * info_k
	return entropy


# 编写函数, 实现按照给定特征划分数据集。
def split_data(data_origin, axis, value):
	"""根据第axis个轴的特征A划分数据，返回A=value的部分，Di
	返回一个
	:param data_origin:
	:param axis:
	:param value:
	:return:
	"""
	data_reduced = []
	for vec in data_origin:
		if vec[axis] == value:
			# 隔开 axis 这一列提取其它列的数据
			# 保存到变量 reducedFeatVec 中
			reduced_vec = vec[:]
			reduced_vec.pop(axis)
			data_reduced.append(reduced_vec)
	return data_reduced


# 实现特征选择函数。遍历整个数据集, 循环计算熵和splitDataSet()函数, 找到最好的特征划分方式。
def choose_feature(data):
	# 获取属性个数,保存到变量 numFeatures
	# 注意数据集中最后一列是分类结果
	feature_count = len(data[0]) - 1
	base_entropy = calc_entropy(data)
	best_info_gain = 0.0
	best_feature = -1
	for i in range(feature_count):
		# 获取数据集中某一属性的所有取值
		feature_list = [sample[i] for sample in data]
		# 获取该属性所有不重复的取值,保存到 uniqueVals 中
		# 可使用 set()函数去重
		unique_val = set(feature_list)

		new_entropy = 0.0
		for value in unique_val:
			sub_data = split_data(data, i, value)
			# 计算按照第 i 列的某一个值分割数据集后的熵
			# 参考文档开始部分介绍的公式
			count_di = len(sub_data)
			prob_di = count_di / len(data)
			info_di = calc_entropy(sub_data)
			new_entropy += prob_di * info_di

		# print('base entropy: ', base_entropy)
		info_gain = base_entropy - new_entropy
		# print('info gain ', labels[i], ': ', info_gain)
		if info_gain > best_info_gain:
			best_info_gain = info_gain
			best_feature = i
	# print(best_feature)
	return best_feature


"""决策树创建过程中会采用递归的原则处理数据集。递归的终止条件为: 
	程序遍历完所有划分数据集的属性;
	或者每一个分支下的所有实例都具有相同的分类。
	如果数据集已经处理了所有属性, 但是类标签依然不是唯一的, 
	此时我们需要决定如何定义该叶子节点, 在这种情况下, 通常会采用多数表决的方法决定分类。"""
def majority_cnt(class_list):
	class_count = {}
	for vote in class_list:
		if vote not in class_count.keys():
			class_count[vote] = 1
		else:
			class_count[vote] += 1
	final_class = max(class_count, key=class_count.get)
	return final_class

# 创建决策树。
def create_decision_tree(data, labels):
	# 获取类别列表,类别信息在数据集中的最后一列
	# 使用变量 classList
	class_list = [vec[-1] for vec in data]
	# 以下两段是递归终止条件
	# 如果数据集中所有数据都属于同一类则停止划分
	# 可以使用 classList.count(XXX)函数获得 XXX 的个数,
	# 然后拿这个数和 classList 的长度进行比较,相等则说明
	# 所有数据都属于同一类,返回该类别即可
	if class_list.count('yes') == len(class_list) or class_list.count('no') == len(class_list):
		return class_list[0]
	# 如果已经遍历完所有属性则进行投票,调用上一步的函数
	# 注意,按照所有属性分割完数据集后,数据集中会只剩下一列,这一列是分类结果
	elif len(data) == 1:
		return majority_cnt(class_list)
	# 调用特征选择函数选择最佳分割属性,保存到 bestFeat
	# 根据 bestFeat 获取属性名称,保存到 bestFeatLabel 中
	best_feature_idx = choose_feature(data)
	best_feature = labels[best_feature_idx]
	# 初始化决策树,可以先把第一个属性填好
	my_tree = {best_feature: {}}
	# 删除最佳分离属性的名称以便递归调用
	del(labels[best_feature_idx])
	# 获取最佳分离属性的所有不重复的取值保存到 uniqueVals
	value_list = [sample[best_feature_idx] for sample in data]
	unique_val = set(value_list)
	for value in unique_val:
	# 复制属性名称,以便递归调用
		sub_label = labels[:]
	# 递归调用本函数生成决策树
		my_tree[best_feature][value] = create_decision_tree(split_data(data, best_feature_idx, value), sub_label)
	return my_tree  # 利用构建好的决策树进行分类。


def classify(input_tree, feature_labels, test_vec):
	# 获取树的第一个节点,即属性名称
	first_str = list(input_tree.keys())[0]
	# 获取该节点下的值
	second_dict = input_tree[first_str]
	# 获取该属性名称在原属性名称列表中的下标
	# 保存到变量 featIndex 中
	# 可使用 index(XXX)函数获得 XXX 的下标
	feature_index = feature_labels.index(first_str)
	# 获取待分类数据中该属性的取值,然后在 secondDict
	# 中寻找对应的项的取值
	# 如果得到的是一个字典型的数据,说明在该分支下还
	# 需要进一步比较,因此进行循环调用分类函数;
	# 如果得到的不是字典型数据,说明得到了分类结果
	for key in second_dict.keys():
		print(type(key), '--', type(test_vec[feature_index]))
		if test_vec[feature_index] == key:
			if isinstance(second_dict[key], dict):
				class_label = classify(second_dict[key], feature_labels, test_vec)
			else:
				class_label = second_dict[key]
	return class_label



"""
	** *延伸拓展 ** *
	一、决策树的存储
	决策树的构建是很耗时的任务, 但用创建好的决策树解决分类问题时可以很快
	完成, 因此为了节省时间, 最好在每次执行分类时直接调用已经构造好的决策树。因
	此延伸出了决策树存储的问题。如有需要可使用以下代码:
"""


# # 存储决策树
# def storeTree(inputTree, filename): pass
#
#
# fw = open(filename,’w’)
# pickle.dump(inputTree, fw)
# fw.close()
#
#
# # 根据文件名读取决策树
# def grabTree(filename):
# 	pass
#
#
# fr = open(filename)
# return pickle.load(fr)


if __name__ == '__main__':
	# data = pd.read_csv('bank.csv').iloc[:1050]
	# bank = data[['marital', 'loan', 'education', 'housing']]
	# label = ['marital', 'loan', 'education', 'housing']

	data, labels = createDataSet()
	id3 = ID3()
	# tree = id3.fit(bank.iloc[:1000].values, label)
	tree = id3.fit(data, labels)
	print(tree)
	# acc = id3.evaluate(bank.iloc[-50:, :-1].values, bank.iloc[-50:, -1].values)
	# print('Accuracy: ', acc)
	create_plot(tree)
