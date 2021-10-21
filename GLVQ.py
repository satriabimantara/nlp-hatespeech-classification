import random as rd
import math
import numpy as np


class GLVQ:

    def __init__(self, max_epoch=1000, alpha=0.9, min_error=0.001):
        self.params = dict()
        self.params["alpha"] = alpha
        self.params["max_epoch"] = max_epoch
        self.params["min_error"] = min_error

    def getParams(self):
        return self.params

    def setParams(self):
        pass

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
        max_epoch = self.params['max_epoch']
        iterator = 0
        self.initializeWeights()
        weights = self.weights
        while alpha > min_error or iterator < max_epoch:
            iterator += 1
            # Looping untuk seluruh data latih yang ada
            for iterate, x in enumerate(self.X_train):
                dist_x_to_weight = []
                # Hitung jarak data ke-x dengan setiap bobot yang ada
                for label, w in weights.items():
                    dist = math.sqrt(sum(w-x)**2)
                    dist_x_to_weight.append(dist)

                if np.argmin(dist_x_to_weight) == y_train[iterate]:
                    index_pemenang = np.argmin(dist_x_to_weight)
                    index_runner = np.argmax(dist_x_to_weight)
                    d1 = dist_x_to_weight[np.argmin(dist_x_to_weight)]
                    d2 = dist_x_to_weight[np.argmax(dist_x_to_weight)]
                else:
                    index_pemenang = np.argmax(dist_x_to_weight)
                    index_runner = np.argmin(dist_x_to_weight)
                    d1 = dist_x_to_weight[np.argmax(dist_x_to_weight)]
                    d2 = dist_x_to_weight[np.argmin(dist_x_to_weight)]

                d_total = d1+d2

                miu_x = ((d1-d2)/(d_total))*-1
                f_x = (1/(1+pow(math.e, (miu_x*iterator))))
                turunan = f_x*(1-f_x)

                weights[index_pemenang] += (
                    alpha * turunan * (d2/pow(d_total, 2)) * (x-weights[index_pemenang]))
                weights[index_runner] -= (
                    alpha * turunan * (d1/pow(d_total, 2)) * (x-weights[index_runner]))

            # perbarui nilai alpha dan iterasi
            alpha = alpha * (1-(iterator/max_epoch))
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
