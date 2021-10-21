import random as rd
import math
import numpy as np


class LVQ:

    def __init__(self, max_epoch=1000, dec_alpha=0.01, alpha=0.5, min_error=0.001):
        self.params = dict()
        self.params["dec_alpha"] = dec_alpha
        self.params["alpha"] = alpha
        self.params["max_epoch"] = max_epoch
        self.params["min_error"] = min_error

    def getParams(self):
        return self.params

    def initializeWeights(self):
        X_train = self.X_train
        y_train = self.y_train
        label, train_index = np.unique(y_train, return_index=True)
        jumlah_kelas = len(train_index)
        iterator = 0
        index_data_train = []
        self.weights = dict()
        while iterator < jumlah_kelas:
            rand_number = rd.randint(0, len(y_train)-1)
            if rand_number not in index_data_train:
                index_data_train.append(rand_number)
                self.weights[iterator] = X_train[rand_number]
                iterator += 1
        self.X_train = np.delete(X_train, index_data_train, axis=0)
        self.y_train = np.delete(y_train, index_data_train, axis=0)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        alpha = self.params['alpha']
        min_error = self.params['min_error']
        dec_alpha = self.params['dec_alpha']
        max_epoch = self.params['max_epoch']
        iterator = 0
        self.initializeWeights()
        weights = self.weights
        while alpha > min_error or iterator < max_epoch:
            # Looping untuk seluruh data latih yang ada
            for iterate, x in enumerate(self.X_train):
                dist_x_to_weight = []
                # Hitung jarak data ke-x dengan setiap bobot yang ada
                for label, w in weights.items():
                    dist = math.sqrt(sum(w-x)**2)
                    dist_x_to_weight.append(dist)

                # tentukan kelas pemenang dengan jarak bobot terpendek dari x
                index_kelas_pemenang = np.argmin(dist_x_to_weight)
                sign = 1 if y_train[iterate] == index_kelas_pemenang else -1

                # update aturan bobot sesuai ketentuan
                weights[index_kelas_pemenang] += sign * \
                    (alpha * (x-weights[index_kelas_pemenang]))
            # perbarui nilai alpha dan iterasi
            iterator += 1
            alpha = alpha - (alpha*dec_alpha)
        self.weights = weights

    def predict(self, X_test):
        dist = []
        for label, w in self.weights.items():
            dist.append(math.sqrt(sum(w-X_test)**2))
        return np.argmin(dist)

    def score(self, X_test, y_test):
        sum_accuracy = 0
        for itr, x in enumerate(X_test):
            distance = []
            for label, w in self.weights.items():
                distance.append(math.sqrt(sum(w-x)**2))
            if np.argmin(distance) == y_test[itr]:
                sum_accuracy += 1
        return sum_accuracy/len(y_test)
