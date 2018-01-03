from global_module.reader import DataReader

class KeywordExtractor:
    def __init__(self, input_file, data_col, label_col):
        self.filename = input_file
        self.data_reader = DataReader(data_col, label_col, input_file)

    def get_unigram_keywords(self):
        vectorizer, matrix = self.data_reader.run_unigram_vectorizer()


    def get_bigram_count_array(self):
        pass