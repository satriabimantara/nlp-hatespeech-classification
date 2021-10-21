import pandas as pd
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Preprocessing:
    def __init__(self):
        pass

    def loadWordNormalize(self):
        data_location = os.getcwd()+"\data\\kata_normalisasi.xlsx"
        normalizad_word = pd.read_excel(data_location)
        return normalizad_word

    def caseFolding(self, dataset):
        dataset['Tweet'] = dataset['Tweet'].str.lower()
        return dataset

    def removing(self, dataset):
        def remove_special_characters(text):
            # remove tab, new line, ans back slice
            text = text.replace('\\t', " ").replace(
                '\\n', " ").replace('\\u', " ").replace('\\', "")
            # remove non ASCII (emoticon, chinese word, .etc)
            text = text.encode('ascii', 'replace').decode('ascii')
            # remove mention, link, hashtag
            text = ' '.join(
                re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
            # remove incomplete URL
            return text.replace("http://", " ").replace("https://", " ")

        def remove_number(text):
            # menghapus angka
            text = re.sub('\w*\d\w*', '', text)
            return text

        def remove_punctuation(text):
            # menghilangkan punctuation
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            return text

        def remove_single_character(text):
            return re.sub(r"\b[a-zA-Z]\b", "", text)

        def remove_whitespace(text):
            text = re.sub('\s+', ' ', text)
            text = text.strip()
            return text

        def get_word_in_brackets(text):
            # mengambil kata di dalam kurung
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('\(.*?\)', '', text)
            return text

        dataset['Tweet'] = dataset['Tweet'].apply(remove_number)
        dataset['Tweet'] = dataset['Tweet'].apply(remove_punctuation)
        dataset['Tweet'] = dataset['Tweet'].apply(remove_single_character)
        dataset['Tweet'] = dataset['Tweet'].apply(remove_special_characters)
        dataset['Tweet'] = dataset['Tweet'].apply(remove_whitespace)
        dataset['Tweet'] = dataset['Tweet'].apply(get_word_in_brackets)
        return dataset

    def tokenizing(self, dataset):
        def tokenize_word(text):
            return word_tokenize(text)

        def freq_dist(text):
            return FreqDist(text)
        dataset['tweet_tokens'] = dataset['Tweet'].apply(tokenize_word)
        dataset['tweet_freq_dist'] = dataset['tweet_tokens'].apply(freq_dist)
        return dataset

    def normalization(self, dataset):
        normalizad_word = self.loadWordNormalize()
        normalizad_word_dict = {}
        for index, row in normalizad_word.iterrows():
            if row[0] not in normalizad_word_dict:
                normalizad_word_dict[row[0]] = row[1]

        def normalized_term(document):
            return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

        dataset['tweet_normalized'] = dataset['tweet_tokens'].apply(
            normalized_term)
        return dataset

    def stopWordRemoval(self, dataset):
        stop_word_list = stopwords.words('indonesian')
        stop_word_list = set(stop_word_list)

        def stop_word_removal(text):
            text = [kata for kata in text if kata not in stop_word_list]
            return text
        dataset['token_stop_word'] = dataset['tweet_normalized'].apply(
            stop_word_removal)
        return dataset

    def stemming(self, tokens_status, dataset):
        def stemming_term(term):
            return stemmer.stem(term)

        def stemming_dataframe(document):
            return [term_dictionary[word] for word in document]

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        term_dictionary = {}
        # inisialisasi dictionary
        for document in dataset[tokens_status]:
            for word in document:
                if word not in term_dictionary:
                    term_dictionary[word] = stemming_term(word)
        dataset['tweet_stemming'] = dataset[tokens_status].apply(
            stemming_dataframe)
        return dataset
