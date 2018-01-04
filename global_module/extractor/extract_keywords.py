from global_module.reader import DataReader
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import pickle


class KeywordExtractor:
    def __init__(self, input_file, data_col, label_col):
        self.filename = input_file
        self.data_reader = DataReader(data_col, label_col, input_file)
        self.id_to_label = self.get_label()
        self.tfidf_transformer = TfidfTransformer(smooth_idf=False)

    def get_label(self):
        label_to_class = {}
        for key in self.data_reader.label_corpus:
            label_to_class[self.data_reader.label_corpus[key]] = key
        return label_to_class

    def _extract_keywords_helper(self, keyindex, keyword, vocab_dict, k, threshold):
        id_to_word = {}
        extracted_keyword = []
        keyword_dict = {}

        for key in vocab_dict:
            id_to_word[vocab_dict[key]] = key

        for idx, each_class in enumerate(keyindex):
            top_keywords_idx = each_class[-k:]
            top_keywords = []
            for each_top_keyword in top_keywords_idx:
                keyword_score = keyword[idx][each_top_keyword]
                if keyword_score >= threshold:
                    curr_keyword = id_to_word[each_top_keyword]
                    top_keywords.append((curr_keyword, keyword_score))
                    if curr_keyword in keyword_dict:
                        keyword_dict[curr_keyword][idx] += keyword_score
                    else:
                        keyword_dict[curr_keyword] = np.zeros(len(keyindex), dtype=np.float32)
                        keyword_dict[curr_keyword][idx] = keyword_score
            extracted_keyword.append(top_keywords)
        return extracted_keyword, keyword_dict

    def extract_keywords(self, count_matrix, vocab_dict, k, threshold):
        vocab_size = len(vocab_dict)

        class_count_matrix = [[0 for _ in range(vocab_size)] for _ in range(len(self.data_reader.label_corpus))]
        for idx, each_instance in enumerate(count_matrix):
            instance_label = self.data_reader.label_corpus[self.data_reader.data_obj.data_instance[idx].text_label]
            class_count_matrix[instance_label] = np.add(each_instance, class_count_matrix[instance_label])
        tfidf = self.tfidf_transformer.fit_transform(class_count_matrix).toarray()
        top_keyindex = tfidf.argsort()
        top_keywords, keyword_dict = self._extract_keywords_helper(top_keyindex, tfidf, vocab_dict, k, threshold)
        return top_keywords, keyword_dict

    def get_unigram_keywords(self, k, threshold):
        vectorizer, matrix = self.data_reader.run_unigram_vectorizer()
        top_keywords, keyword_dict = self.extract_keywords(matrix, vectorizer.vocabulary_, k, threshold)
        with open(self.filename + '_unigram.pickle', 'wb') as handle:
            pickle.dump(keyword_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return vectorizer, matrix, top_keywords

    def get_bigram_keywords(self, k, threshold):
        vectorizer, matrix = self.data_reader.run_bigram_vectorizer()
        top_keywords, keyword_dict = self.extract_keywords(matrix, vectorizer.vocabulary_, k, threshold)
        with open(self.filename + '_bigram.pickle', 'wb') as handle:
            pickle.dump(keyword_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return vectorizer, matrix, top_keywords


def main():
    extractor = KeywordExtractor('filepath', 1, 0)

    k = 10
    threshold = 0.2
    vectorizer, matrix, top_keywords = extractor.get_unigram_keywords(k, threshold)

    vectorizer, matrix, top_keywords = extractor.get_bigram_keywords(k, threshold)


if __name__ == '__main__':
    main()
