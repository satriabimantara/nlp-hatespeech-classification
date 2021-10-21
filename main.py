from Preprocessing import Preprocessing
from Utils import Utils
from Menu import Menu
from Termweighting import TFIDF
from Pengujian import Pengujian

# instansisasi objek
prep = Preprocessing()
menu = Menu()
utils = Utils()
tfidf = TFIDF()
pengujian = Pengujian()


def preprocessingTweet(dataset, num_pengujian):
    print("Preprocessing sedang berlangsung...")
    print("Mohon tunggu sebentar...")
    print("")
    dataset = prep.caseFolding(dataset)
    dataset = prep.removing(dataset)
    dataset = prep.tokenizing(dataset)
    dataset = prep.normalization(dataset)
    if num_pengujian == 3:
        tokens_status = "tweet_normalized"
    else:
        dataset = prep.stopWordRemoval(dataset)
        tokens_status = "token_stop_word"
    dataset = prep.stemming(tokens_status, dataset)
    utils.saveCsvFile(dataset, "data_preprocessing.csv")


def documentTermMatrix():
    dataset = utils.loadData(
        "data_preprocessing.csv", usecolumns=['Label', 'tweet_stemming'])
    dataset = utils.changeColumnsName(dataset, ['Label', "Tweet"])
    dataset = tfidf.termWeighting(dataset)
    utils.saveCsvFile(dataset, "term_document.csv")


def main():
    num_pengujian, amount = menu.main()
    # Load data
    dataset = utils.loadData("tweet.csv")
    dataset = utils.replaceAttributeDataframe(
        dataset, "Label", {"Non_HS": "0", "HS": "1"})

    if num_pengujian != 1:
        dataset = utils.samplingData(dataset)

    # Preprocessing Tweet
    datasetAfterPreprocessing = preprocessingTweet(dataset, num_pengujian)

    # Termweighting
    documentTermMatrix()

    # load data after document term matrix
    dataset = utils.loadData("term_document.csv")

    # Pengujian
    utils.showAlert(['Proses pengujian sedang dilakukan...',
                     'Proses ini akan memakan waktu beberapa menit...', 'Mohon tunggu sebentar...'])
    for i in range(1):
        print("+++"*30)
        print("HASIL PENGUJIAN KE-{}".format(i+1))
        print("+++"*30)
        pengujian.transform(dataset)
        pengujian.score()


if __name__ == "__main__":
    main()
