# -*- coding: utf-8 -*-
"""
Created on 19-4-28 下午11:04

@author: chinbing <x62881999@gmail.com>
"""

import jieba
# import jieba.analyse
import re

path = '中文情感分析语料库/hotel/'


class Tokenizer:
	def __init__(self, stopwords=[]):
		self.stopwords = stopwords

	def isfloat(self, s):
		regex = re.compile('^(-?\d+)(\.\d*)?$')
		if regex.match(s):
			return True
		return False

	def _cut(self, sentence):
		words = jieba.cut(sentence)
		words = [word for word in words if word not in self.stopwords]
		words = [word for word in words if word is not ' ' and not word.isdigit() and not self.isfloat(word)][:-1][:30]

		# tags是使用基于TF-IDF算法的关键词抽取来分词的版本，但是因为直观上有关于情感的词好像这个值不高，所以这个方案最后被我舍弃了
		# tags = jieba.analyse.extract_tags(sentence, 30)
		# tags = [tag for tag in tags if not tag.isdigit() and not self.isfloat(tag)]
		return words


if __name__ == '__main__':
	# jieba.analyse.set_stop_words(path+'chinese_stopwords.txt')
	stopwords = [word.strip() for word in open(path+'chinese_stopwords.txt').readlines()]
	tokenizer = Tokenizer(stopwords)

	f = open('negative').readlines()

	words_pos = open('words_neg', 'w')
	for sentence in f:
		words = tokenizer._cut(sentence)
		print(','.join(words), file=words_pos)
	words_pos.close()




