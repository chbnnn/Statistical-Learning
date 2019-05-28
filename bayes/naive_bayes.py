# -*- coding: utf-8 -*-
"""
Created on 19-4-10 下午8:18

@author: chinbing <x62881999@gmail.com> (原创声明)
"""
import numpy as np
from sklearn.metrics import average_precision_score, recall_score, f1_score
from tqdm import tqdm, trange
from tokenizer import Tokenizer
import sys
import os

class NaiveBayes:
	"""
	朴素贝叶斯分类器
	"""
	def __init__(self, tokenizer, dataset=None, vocabulary=[]):
		"""
		:param dataset: 可选接收参数dataset为用于生成词汇表的语料，默认为None
		:param vocabulary: 可选接受参数vocabulary，直接指定词汇表，默认为[]
		"""
		self.tokenizer = tokenizer
		self.vocabulary = vocabulary
		self.pc0 = None
		self.pc1 = None
		self.prob_1 = 0.5
		if dataset is not None:
			self._create_vocabulary(dataset)

	def _create_vocabulary(self, sentences):
		print('----building vocabulary----')
		vocabulary = set()
		for sentence in tqdm(sentences):
			sentence = sentence.strip().split(',')
			vocabulary = vocabulary | set(sentence)  # 取两个集合的并集
		self.vocabulary = list(vocabulary)
		print('----vocabulary built----')

	# 构建词向量。这里采用的是词集模型，即只需记录每个词是否出现，而不考虑其出现的次数。需要记录词出现的次数的叫词袋模型。
	def _sentence_to_vector(self, sentence, is_str=True):
		vec = np.zeros(len(self.vocabulary))  # 生成零向量的array
		if is_str:
			for word in sentence.strip().split(','):
				if word in self.vocabulary:
					vec[self.vocabulary.index(word)] = 1  # 单词出现则记为1
				else:
					print('the word: %s is not in my vocabulary!' % word)
		else:
			for word in sentence:
				if word in self.vocabulary:
					vec[self.vocabulary.index(word)] = 1  # 单词出现则记为1
				else:
					print('the word: %s is not in my vocabulary!' % word)
		return vec  # 返回全为0或1的向量

	# 在实现算法时，需要考虑两个问题：
	# 	a.当使用连乘计算时，若某一个词的概率为0，那么最终的结果也会为0，这是不正确的。为防止这种情况，需要将所有词项出现的次数都初始化为1，每一类所有词项数量初始化为2；
	# 	b.在连乘时，为防止单项概率过小导致连乘结果下溢，需要对结果求自然对数将其转化为加法
	def fit(self, x_train, y_train, is_raw=True):
		"""
		训练，数据拟合过程
		:param x_train:
		:param y_train:
		:param is_raw: 是否为句子，False表示已经转化为向量
		:return:
		"""
		print('----start training----')
		print('  --converting sentences to vectors')
		# 如果还未转化为向量，则通过查表转化为下标向量
		if is_raw:
			x = []
			for sentence in tqdm(x_train):
				x.append(self._sentence_to_vector(sentence))
			x_train = x
		print('  --sentences converted, start training')

		self.p_1 = y_train.count(1) / len(y_train)

		pos, neg = [], []
		m = len(y_train)
		for _ in trange(m):
			i = np.random.randint(0, m)
			sentence = np.array(x_train[i])
			if y_train[i]:
				pos += sentence.nonzero()[0].tolist()
			else:
				neg += sentence.nonzero()[0].tolist()

		m = len(x_train[0])
		self.pc1, self.pc0 = np.zeros(m), np.zeros(m)

		for i in range(m):
			self.pc1[i] += np.log((pos.count(i) + 1) / (len(pos) + m))
			self.pc0[i] += np.log((neg.count(i) + 1) / (len(neg) + m))
		print('----training finished----')

	def classify(self, vec):
		"""
		分类单个向量
		:param vec: 欲分类的向量
		:return: 类别，0或1
		"""
		if isinstance(vec, str):
			vec = self._sentence_to_vector(vec)

		p1 = sum(vec * self.pc1) + np.log(self.prob_1)
		p0 = sum(vec * self.pc0) + np.log(1 - self.prob_1)
		return 1 if p1 > p0 else 0

	def predict(self, sentence):
		"""
		预测数据集，对整个数据集进行分类，但不对分类结果进行评估u
		:param x_test:
		:return: 无，直接打印分类结果
		"""
		words = self.tokenizer._cut(sentence)
		vec = self._sentence_to_vector(words, is_str=False)
		print('分类结果: ', self.classify(vec))

	def evaluate(self, x_test, y_test):
		"""
		分类整个测试集，并与正确结果对比进行评估
		:param x_test:
		:param y_test:
		:return: acc, rec, f1
		"""
		print('----evaluation started----')
		y_ = []
		for sentence in x_test:
			y_.append(self.classify(sentence))

		y_ = np.array(y_).reshape([-1, 1])
		y_test = np.array(y_test).reshape([-1, 1])

		acc = average_precision_score(y_test, y_)
		rec = recall_score(y_test, y_)
		f1 = f1_score(y_test, y_)
		print('----evaluation done----')
		return acc, rec, f1


if __name__ == '__main__':
	# 分好词并去掉停止词的正面与负面评价，各5000条
	with open('words_pos') as words_pos, open('words_neg') as words_neg:
		pos, neg = words_pos.readlines(), words_neg.readlines()

		stopwords = [word.strip() for word in open('chinese_stopwords.txt').readlines()]
		tokenizer = Tokenizer(stopwords)
		# 封装的朴素贝叶斯分类器，接受参数为用来生成词汇表的语料
		# 这里将训练与测试数据一起用来生成词汇表，防止测试集出现词汇表外的单词的情况
		naive_bayes = NaiveBayes(tokenizer, pos + neg)

		# 测试集保留正面负面评价各500条
		x_test = pos[-500:] + neg[-500:]
		y_test = [1] * 500 + [0] * 500

		# 其余全作为训练集
		x_train = pos[:4500] + neg[:4500]
		y_train = [1] * 4500 + [0] * 4500
		naive_bayes.fit(x_train, y_train)

		# 三折交叉验证，但是因为训练数据只有4500条，分为三折，训练集只分配到3000条，降低了最后在测试集的表现，所以最终决定不用三折交叉验证
		# P, R, F1 = [], [], []
		# for epoch in range(3):
		# 	print('--epoch ', epoch)
		# 	x_dev = pos[1500*epoch : 1500*epoch + 1500] + neg[1500*epoch : 1500*epoch + 1500]
		# 	x_train = pos[0 : 1500*epoch] + pos[1500*epoch + 1500 : ] +  neg[0 : 1500*epoch] + neg[1500*epoch + 1500 : ]
		# 	y_dev = [1] * 1500 + [0] * 1500
		# 	y_train = [1] * 3000 + [0] * 3000
		# 	naive_bayes.fit(x_train, y_train)
		# 	acc, rec, f1 = naive_bayes.evaluate(x_dev, y_dev)
		# 	P.append(acc)
		# 	R.append(rec)
		# 	F1.append(f1)
		#
		# print('\n----cross validation result----')
		# print('average precision value: ', np.mean(P))
		# print('average recall value: ', np.mean(R))
		# print('average f1 value: ', np.mean(F1))
		print('请选择操作： 0 测试；1 显示测试集结果 ', end='')
		param = input() if len(sys.argv) < 2 else sys.argv[1]
		if param in ['eval', '1']:
			acc, rec, f1 = naive_bayes.evaluate(x_test, y_test)
			print('\n----test set performance----')
			print('  precision: ', acc)
			print('  recall: ', rec)
			print('  f1 score: ', f1)
		elif param in ['test', '0']:
			while True:
				sentence = input('输入酒店评价(输入#退出测试): ')
				if sentence == '#':
					break
				naive_bayes.predict(sentence)

		print('原创声明- -')
		input('press any key to exit...')




