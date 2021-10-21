import os
import random as rd
import pandas as pd


class Utils:
    def __init__(self):
        pass

    def loadData(self, dataName, usecolumns=[]):
        data_location = os.getcwd()+"\data\\"+dataName
        if len(usecolumns) == 0:
            dataset = pd.read_csv(data_location)
        else:
            dataset = pd.read_csv(data_location, usecols=usecolumns)
        return dataset

    def replaceAttributeDataframe(self, dataset, atributName, value):
        dataset[atributName].replace(value, inplace=True)
        return dataset

    def changeColumnsName(self, dataset, new_columns):
        dataset.columns = dataset.columns = ['Label', 'Tweet']
        return dataset

    def saveCsvFile(self, dataset, name_file):
        location = os.getcwd()+"\data\\"+name_file
        dataset.to_csv(location, index=False)

    def splitAttributesAndTarget(self, dataset, targetName):
        attributes = dataset.drop(targetName, axis=1)
        target = dataset[targetName]
        return attributes, target

    def showAlert(self, message):
        print("")
        for msg in message:
            print(msg)

    def samplingData(self, dataset):
        # Pisahkan target dan atribut
        target = dataset['Label'].to_numpy()
        tweet = dataset['Tweet'].to_numpy()

        new_tweet = []
        new_target = []
        max_sampling = 260
        # sampling for NON-HS
        index_random_non_hs = []
        index_random_hs = []
        while len(new_tweet) < max_sampling:
            rand_number = rd.randint(0, 452)
            if not (rand_number in index_random_non_hs):
                new_tweet.append(tweet[rand_number])
                new_target.append(target[rand_number])
                index_random_non_hs.append(rand_number)
        # masukkan sisa dari sample HS
        for i in range(453, len(dataset)):
            new_tweet.append(tweet[i])
            new_target.append(target[i])

        # make dataframe from numpy
        new_df = pd.DataFrame(new_tweet, columns=['Tweet'])
        new_df['Label'] = new_target
        dataset = new_df
        return dataset
