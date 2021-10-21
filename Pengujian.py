import numpy as np
from Utils import Utils
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from GLVQ import GLVQ
from LVQ import LVQ
import operator


class Pengujian:
    def __init__(self):
        self.params_lvq = dict()
        self.params_lvq['n_components_svd'] = [1]
        self.params_lvq['alphas'] = [i for i in np.arange(0.1, 1.1, 0.1)]
        self.params_lvq['dec_alphas'] = [0.1, 0.01]
        self.params_lvq['max_epochs'] = [100]

        self.params_glvq = dict()
        self.params_glvq['n_components_svd'] = self.params_lvq['n_components_svd']
        self.params_glvq['alphas'] = self.params_lvq['alphas']
        self.params_glvq['max_epochs'] = [100, 150]

        self.score_LVQ = dict()
        self.scores_GLVQ = dict()
        self.utils = Utils()

    def transform(self, dataset):
        attributes, target = self.utils.splitAttributesAndTarget(
            dataset, "Label")
        kfold = StratifiedKFold(n_splits=10)
        n_components_SVD = self.params_lvq['n_components_svd']
        alphas = self.params_glvq['alphas']
        dec_alphas = self.params_lvq['dec_alphas']
        max_epochs_glvq = self.params_glvq['max_epochs']
        max_epochs_lvq = self.params_lvq['max_epochs']

        for n_components in n_components_SVD:
            svd = TruncatedSVD(n_components=n_components)
            attributes_reducted = svd.fit_transform(attributes)
            np_attributes = attributes_reducted
            np_target = target.to_numpy()

            for alpha in alphas:
                # pengujian GLVQ
                for max_epoch in max_epochs_glvq:
                    key = "SVDComponent-" + \
                        str(n_components)+"_alpha-"+str(alpha) + \
                        "_maxepoch-"+str(max_epoch)
                    model_glvq = GLVQ(max_epoch=max_epoch, alpha=alpha)
                    score_folds_glvq = []
                    for train_index, test_index in kfold.split(np_attributes, np_target):
                        X_train = np_attributes[train_index]
                        X_test = np_attributes[test_index]
                        y_train = np_target[train_index]
                        y_test = np_target[test_index]
                        model_glvq.fit(X_train, y_train)
                        score_folds_glvq.append(
                            model_glvq.score(X_test, y_test))
                    self.scores_GLVQ[key] = sum(
                        score_folds_glvq)/len(score_folds_glvq)

                # Pengujian LVQ
                for max_epoch in max_epochs_lvq:
                    for dec_alpha in dec_alphas:
                        key = "SVDComponent-" + \
                            str(n_components)+"_alpha-"+str(alpha) + \
                            "_maxepoch-"+str(max_epoch) + \
                            "_decalpha-"+str(dec_alpha)
                        model_lvq = LVQ(max_epoch=max_epoch,
                                        dec_alpha=dec_alpha, alpha=alpha)
                        score_folds_lvq = []
                        for train_index, test_index in kfold.split(np_attributes, np_target):
                            X_train = np_attributes[train_index]
                            X_test = np_attributes[test_index]
                            y_train = np_target[train_index]
                            y_test = np_target[test_index]
                            model_lvq.fit(X_train, y_train)
                            score_folds_lvq.append(
                                model_lvq.score(X_test, y_test))
                        self.score_LVQ[key] = sum(
                            score_folds_lvq)/len(score_folds_lvq)

    def score(self):
        print("=========")
        print("MODEL LVQ")
        print("=========")
        print("Hyperparameter Terbaik : {}".format(max(self.score_LVQ.items(), key=operator.itemgetter(
            1))[0]))
        print("Accuracy Tertinggi : {}".format(
            self.score_LVQ[max(self.score_LVQ.items(), key=operator.itemgetter(1))[0]]))
        print()
        print("--- Hasil Detail ---")
        for key, accuracy in self.score_LVQ.items():
            print("{} : {}".format(key, accuracy))

        print("")
        print("=========")
        print("MODEL GLVQ")
        print("=========")
        print("Hyperparameter Terbaik : {}".format(max(self.scores_GLVQ.items(), key=operator.itemgetter(
            1))[0]))
        print("Accuracy Tertinggi : {}".format(
            self.scores_GLVQ[max(self.scores_GLVQ.items(), key=operator.itemgetter(1))[0]]))
        print()
        print("--- Hasil Detail ---")
        for key, accuracy in self.scores_GLVQ.items():
            print("{} : {}".format(key, accuracy))
