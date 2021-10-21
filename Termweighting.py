import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    def __init__(self):
        pass

    def termWeighting(self, dataset):
        def join_text_list(texts):
            texts = ast.literal_eval(texts)
            return ' '.join([text for text in texts])

        tfidf = TfidfVectorizer(smooth_idf=False, binary=True, norm=None)
        dataset["tweet_join"] = dataset["Tweet"].apply(join_text_list)
        tfidf_mat = tfidf.fit_transform(dataset['tweet_join']).toarray()
        new_dataset = pd.DataFrame(
            tfidf_mat, columns=tfidf.get_feature_names())
        new_dataset['Label'] = dataset['Label']
        return new_dataset
