# 创建一个名为bayes.py的文件，将下面出现的代码及自己编写的代码均放入该文件中。

import numpy as np

# ①准备数据。使用提供的函数：
def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
									['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
									['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
									['stop', 'posting', 'stupid', 'worthless', 'garbage'],
									['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
									['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

	classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言论，0表示正常言论

	return postingList, classVec


# ②构建词汇表生成函数：
def createVocabList(dataSet):
	vocabSet = set()
	for document in dataSet:
		vocabSet = vocabSet | set(document)  # 取两个集合的并集
	return list(vocabSet)


# ③构建词向量。这里采用的是词集模型，即只需记录每个词是否出现，而不考虑其出现的次数。需要记录词出现的次数的叫词袋模型。
def setOfWords2Vec(vocabList, inputSet):
	returnVec = np.zeros(len(vocabList))  # 生成零向量的array
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1  # 单词出现则记为1
		else:
			print('the word: %s is not in my Vocabulary!' % word)
	return returnVec  # 返回全为0和1的向量


# ④根据训练集计算概率。根据上面的公式：
#
# 正如前面分析的那样，我们只需考虑分子即可。
# 用训练集中，属于类别的样本数量除以总的样本数量即可；
# 可以根据前面的独立性假设，先分别计算，，等项，再将其结果相乘即可，而的计算公式为：
#
# 在实现算法时，需要考虑两个问题：
# 	a.当使用连乘计算时，若某一个词的概率为0，那么最终的结果也会为0，这是不正确的。为防止这种情况，需要将所有词项出现的次数都初始化为1，每一类所有词项数量初始化为2；
# 	b.在连乘时，为防止单项概率过小导致连乘结果下溢，需要对结果求自然对数将其转化为加法，因为。
def trainNB(trainMat, listClasses):
	p1 = listClasses.count(1) / len(listClasses)

	pos, neg = [], []
	for i, c in enumerate(listClasses):
		sentence = np.array(trainMat[i])
		if c:
			pos += sentence.nonzero()[0].tolist()
		else:
			neg += sentence.nonzero()[0].tolist()

	m = len(trainMat[0])
	pc1, pc0 = np.zeros(m), np.zeros(m)

	for i in range(m):
		pc1[i] += np.log((pos.count(i) + 1) / (len(pos) + m))
		pc0[i] += np.log((neg.count(i) + 1) / (len(neg) + m))

	return pc0, pc1, p1


# ⑤根据上一步计算出来的概率编写分类器函数。
# 其中，p0Vec，p1Vec，pClass1均为上一步函数的返回值，分别代表公式中的，以及p(c1)。
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0


# ⑥编写测试函数。
def testingNB():
	listPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listPosts)

	trainMat = []

	for postinDoc in listPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

	p0V, p1V, pAb = trainNB(trainMat, listClasses)

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = setOfWords2Vec(myVocabList, testEntry)
	print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


# 在python命令提示符中输入以下语句进行测试：
# import bayes
if __name__ == '__main__':
	testingNB()
