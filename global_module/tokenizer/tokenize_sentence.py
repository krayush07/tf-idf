# -*- coding: utf-8 -*-
import nltk

class SentenceTokenizer:
    def tokenize(self, sentence):
        return str(' '.join(nltk.word_tokenize(sentence))).strip()