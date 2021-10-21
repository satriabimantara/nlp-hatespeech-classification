class Menu:

    def __init__(self):
        pass

    def intInput(self, messagePrompt, messageError):
        while True:
            try:
                num = int(input(messagePrompt))
                return num
            except ValueError as e:
                print(messageError)

    def inputAmountPengujian(self):
        num = self.intInput("Masukkan jumlah pengujian >> ",
                            "Anda memasukkan selain angka!")
        return num

    def main(self):
        print("")
        print("=="*15)
        print("Aplikasi Klasifikasi Teks Ujaran Kebencian Berbahasa Indonesia Menggunakan Algoritma LVQ dan GLVQ")
        print("=="*15)
        print("1. Pengujian Imbalanced Dataset")
        print("2. Pengujian Balanced Dataset (Random Sampling)")
        print("3. Pengujian Tanpa Tahap Stopword")
        print("4. Pengujian dengan Tahap Stopword")
        print("5. Pengujian dengan Reduksi Dimensi Data")
        num = self.intInput("Masukkan nomor menu >> ",
                            "Anda memasukkan selain angka!")
        amount = self.inputAmountPengujian()
        return num, amount
