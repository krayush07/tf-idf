from sklearn.feature_extraction.text import CountVectorizer
from global_module.tokenizer import SentenceTokenizer


class DataInstance:
    def __init__(self, raw_text=None, tokenized_text=None, text_label=None):
        self.raw_text = raw_text
        self.tokenized_text = tokenized_text
        self.text_label = text_label


class Data:
    def __init__(self, data_col, class_col, input_file):
        self.data_col = data_col
        self.label_col = class_col
        self.data_instance = []
        self._load_data(input_file)

    def _load_data(self, input_file):
        instances = open(input_file, 'r')
        sent_tokenizer = SentenceTokenizer()
        for each_instance in instances:
            instance_split = each_instance.strip().split('\t')
            text = instance_split[self.data_col]
            label = instance_split[self.label_col]
            tokenized_data = sent_tokenizer.tokenize(text).lower()
            self.data_instance.append(DataInstance(text, tokenized_data, label))
        return self.data_instance


class DataReader:
    def __init__(self, data_col, label_col, input_file):
        self.data_obj = Data(data_col, label_col, input_file)
        self._create_data_corpus()

    def _create_data_corpus(self):
        self.data_corpus = []
        self.label_corpus = {}
        for each_instance in self.data_obj.data_instance:
            self.data_corpus.append(each_instance.tokenized_text)
            if each_instance.text_label not in self.label_corpus:
                self.label_corpus[each_instance.text_label] = len(self.label_corpus)

    def run_unigram_vectorizer(self):
        unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
        unigram_matrix = unigram_vectorizer.fit_transform(self.data_corpus).toarray()
        return unigram_vectorizer, unigram_matrix

    def run_bigram_vectorizer(self):
        bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=1)
        bigram_matrix = bigram_vectorizer.fit_transform(self.data_corpus).toarray()
        return bigram_vectorizer, bigram_matrix

    def run_custom_ngram_vectorizer(self, ngram_range):
        self.custom_ngram_vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=1)
        self.custom_ngram_matrix = self.custom_ngram_vectorizer.fit_transform(self.data_corpus)

    def skipgram_vectorizer(self):
        pass
