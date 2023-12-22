from collections import defaultdict
from easyocr import easyocr
from ultralytics import YOLO
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import numpy as np
import cv2
import os
import sys


class ShowImage(QMainWindow):
    # Dijalankan saat objek ShowImage dibuat dan melakukan inisialisasi
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        # self.video_path = './dago.mp4'
        self.video_path = "./Parkiran GD2.mp4"
        self.video = cv2.VideoCapture(self.video_path)

    def start(self):
        # menginisialisasi library easyocr untuk membaca karakter dari hasil crop plat nomor
        reader = easyocr.Reader(['en'], gpu=False)

        # sebuah dictionary yang akan merubah sebuah karakter menjadi angka jika easy ocr malah mendeteksi karakter
        dict_char_to_int = {'O': '0',
                            'Q': '0',
                            'I': '1',
                            'J': '3',
                            'A': '4',
                            'G': '6',
                            'S': '5'}

        # sebuah dictionary yang akan merubah sebuah angka menjadi karakter jika easy ocr malah mendeteksi angka
        dict_int_to_char = {'0': 'O',
                            '1': 'I',
                            '3': 'J',
                            '4': 'A',
                            '6': 'G',
                            '5': 'S',
                            '8': 'B'}

        # sebuah dictionary yang akan merubah sebuah angka/karakter menjadi karakter jika easy ocr salah mendeteksi
        dict_awal = {'0': 'D',
                     'U': 'D',
                     'I': 'D',
                     '4': 'A',
                     '6': 'G',
                     '5': 'S',
                     '8': 'B'}

        # memuat model yang sudah di training dari dataset yang ada
        model = YOLO('best.pt')

        # memasukkan video yang telah diinisialisasi menjadi sebuah variabel bernama cap
        cap = self.video

        # mendapatkan nama file dari video yang dipilih
        filename, file_extension = os.path.splitext(os.path.basename(self.video_path))

        # mengatur format untuk menyimpan video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # menyimpan semua frame hasil pengolahan nanti ke variabel out menjadi sebuah video
        out = cv2.VideoWriter(f'./video/{filename}_out.mp4', fourcc, fps, (width, height))

        # untuk menyimpan semua hasil tracking dari objek
        track_history = defaultdict(lambda: [])

        # menginisiasi variabel yang akan menandakan frame apa saat ini yang diolah
        frame_count = 0

        # mengetahui jumlah frame yang ada pada video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # untuk menyimpan hasil deteksi frame sebelumnya
        prev_license_plate_info = {'plat': 0, 'isi': 'TIDAK TERDETEKSI'}

        # melakukan perulangan selama video masih terbuka
        while cap.isOpened():

            # menyimpan frame saat ini dan status apakah video sudah selesai apa belum
            success, frame = cap.read()

            # masuk ke kondsi jika video belum selesai
            if success:

                # menyimpan frame saat ini ke sebuah image dan ditampilkan pada label original
                self.Image = frame
                self.displayImage(self.label_original)

                # memanfaatkan YOLO untuk melakukan tracking berdasarkan model yang telah di inisiasi sebelumnya
                results = model.track(frame, persist=True)

                # untuk melakukan plotting dari hasil tracking
                annotated_frame = results[0].plot()

                # menampilkan hasil tracking plat beserta id dan conf nya pada label proses pertama
                self.Image = annotated_frame
                self.displayImage(self.label_p1)

                # untuk mendapatkan bounding boxes dari hasil deteksi plat
                boxes = results[0].boxes.xywh.cpu()

                # mengecek apakah plat memiliki id atau tidak
                if hasattr(results[0].boxes.id, 'int'):

                    # jika ada maka id tracking akan dimasukkan ke variabel track_id
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    # melakukan perulangan untuk boounding boxes, track_id dalam zip yang ada
                    for box, track_id in zip(boxes, track_ids):

                        # membagi bounding boxes menjadi x,y yaitu titik tengah objek, dan w/lebar dan h/tinggi
                        x, y, w, h = box

                        # mengambil sebuah daftar track dari track_history dengan menggunakan track_id sebagai indeks
                        track = track_history[track_id]

                        # memperbaru riwayat gerakan objek dengan track_id yang sama
                        track.append((float(x), float(y)))

                        # memeriksa apakah daftar track melebihi 30, jika ya,
                        # maka elemen pertama dari daftar (track.pop(0)) akan dihapus
                        if len(track) > 30:
                            track.pop(0)

                        # mengubah titik tengah objek menjadi titik kiri atas dari objek
                        x_tl = int(x - w / 2)
                        y_tl = int(y - h / 2)
                        x_tl = max(0, x_tl)
                        y_tl = max(0, y_tl)
                        x_tl = int(x_tl)
                        y_tl = int(y_tl)

                        # mengcrop citra menjadi hanya citra platnya saja
                        img_plate_crop = frame[y_tl:y_tl + int(h), x_tl:x_tl + int(w)]
                        img = img_plate_crop.copy()

                        # menampilkan hasil crop plat pada label proses 2
                        self.Image = img
                        self.displayImage(self.label_p2)

                        # mengubah citra hasil crop menjadi citra greyscale
                        img_plate_gray = cv2.cvtColor(img_plate_crop, cv2.COLOR_BGR2GRAY)

                        # melakukan proses resizing
                        img_plate_gray = cv2.resize(img_plate_gray, (
                            int(img_plate_gray.shape[1] * 4), int(img_plate_gray.shape[0] * 4)))

                        # menampilkan citra grayscale dan resize pada label proses 3
                        self.Image = img_plate_gray
                        self.displayImage(self.label_p3)

                        # melakukan proses sharpening untuk mengatasi citra hasil crop yang blur
                        kernel = np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]])
                        sharpened_image = cv2.filter2D(img_plate_gray, -1, kernel)

                        # menampilkan citra hasil sharpening pada label proses 4
                        self.Image = sharpened_image
                        self.displayImage(self.label_p4)

                        # mengubah citra menjadi hitam putih dengan menggunakan otsu thresholding
                        _, img_plate_bw = cv2.threshold(sharpened_image, 0, 255,
                                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        # menampilkan citra hasil otsu pada label proses 5
                        self.Image = img_plate_bw
                        self.displayImage(self.label_p5)

                        # melakukan opening untuk menyambungkan objek yang terputus putus
                        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                        img_plate_bw = cv2.morphologyEx(img_plate_bw, cv2.MORPH_OPEN, kernel)

                        # menampilkan citra hasil opening
                        self.Image = img_plate_bw
                        self.displayImage(self.label_p6)

                        # mendeteksi karakter yang ada pada citra menggunakan reader yang diinisiasi di awal
                        detections = reader.readtext(img_plate_bw)

                        # menginisiasi variabel-variabel relevan
                        text = ''
                        jumlah = 0
                        dua_plat = False
                        full = False

                        # melakukan perulangan berdasarkan jumlah karakter yang terdeteksi
                        for detection in detections:

                            # mendapatkan bounding box, isi text, dan conf score dari hasil deteksi
                            bbox, text_, score = detection

                            # menghilangkan whitespace pada karakter yang terdeteksi
                            text_ = text_.upper().replace(' ', '')

                            # untuk mengetahui berapa banyak sequence karakter yang terdeteksi
                            jumlah += 1

                            # biasanya library easyocr akan mendeteksi huruf pertama/bagian kiri plat terlebih dahulu
                            if jumlah == 1:
                                # mengseleksi huruf pertama diproses apabila = 1 (misal D)
                                # atau = 2 (misal AB)
                                # atau jika tidak ada yang terdeteksi maka secara otomatis akan menjadi D
                                # atau terkadang ada yang terdeteksi random namun diakhirnya benar adalah isi platnya,
                                # maka diambil indeks terakhir dari semua karakternya
                                if len(text_) == 1:
                                    text += text_
                                elif len(text_) == 2:
                                    text += text_
                                    dua_plat = True
                                else:
                                    if text_ == '':
                                        text += 'D'
                                    else:
                                        text += text_[-1]

                            # selanjutnya kondisi ini untuk 4 karakter yang ada pada tengah plat (berupa 4 nomor)
                            elif jumlah == 2:
                                # jika yang terdeteksi 4, maka akan ditambahkan ke array akhir
                                # namun jika hasil deteksi melebih 7 (terkadang 4 nomor dan 2/3 huruf diakhir menyatu)
                                # maka akan diambil 7 huruf pertama
                                # selain itu juga kadang tidak hanya 4 nomor namun ada karakter random yang terdeteksi
                                # maka hanya akan diambil 4 buah karakter pertama pada array
                                if len(text_) == 4:
                                    text += text_
                                elif len(text_) >= 7:
                                    text += text_[:7]
                                    full = True
                                else:
                                    text += text_[:4]

                            # yang terakhir hanya mendeteksi 2/3 karakter pada akhir plat
                            elif jumlah == 3:
                                # jika saat ini plat tidak full dan panjang karakternya dua atau 3, maka akan
                                # ditambahkan ke array akhir
                                # namun jika tidak full maka akan diambil 3 buah huruf terakhir
                                if not full and len(text_) == 2 or len(text_) == 3:
                                    text += text_
                                elif not full:
                                    text += text_[:3]

                        # selanjutnya proses mapping adalah untuk mengubah huruf yang terdeteksi sebagai angka
                        # begitupun sebaliknya
                        license_plate_ = ''
                        if len(text) == 7:
                            mapping = {0: dict_awal, 5: dict_int_to_char, 6: dict_int_to_char,
                                       1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int,
                                       4: dict_char_to_int}
                        elif len(text) == 8:
                            if not dua_plat:
                                mapping = {0: dict_awal, 5: dict_int_to_char, 6: dict_int_to_char,
                                           7: dict_int_to_char,
                                           1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int,
                                           4: dict_char_to_int}
                            else:
                                mapping = {0: dict_awal, 1: dict_int_to_char, 6: dict_int_to_char,
                                           7: dict_int_to_char,
                                           2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int,
                                           5: dict_char_to_int}
                        elif len(text) == 9:
                            mapping = {0: dict_awal, 1: dict_int_to_char, 6: dict_int_to_char,
                                       7: dict_int_to_char,
                                       8: dict_int_to_char,
                                       2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int,
                                       5: dict_char_to_int}
                        else:
                            mapping = None

                        # selanjutnya apabila sudah dilakukan mapping maka akan ditambahkan ke variabel licenses_plate_
                        if mapping is not None:
                            for j in range(0, len(text)):
                                if text[j] in mapping[j].keys():
                                    license_plate_ += mapping[j][text[j]]
                                else:
                                    license_plate_ += text[j]

                        # selanjutnya mengecek apakah format plat yang didapatkan sudah benar
                        # apabila 7 karakter, maka huruf pertama huruf, huruf kedua-kelima angka, dan huruf keenam-
                        # ketujuh adalah huruf, maka memenuhi, contoh = D1234VA
                        # apabila 8 karakter, contoh yang memenuhi adalah D1234ABC atau AB1234CA
                        # apabila 9 karakter, contoh yang memenuhi adalah AB1235ABC
                        plat = False
                        if len(license_plate_) == 7:
                            if license_plate_[0].isalpha() and license_plate_[1:5].isdigit() and \
                                    license_plate_[5:].isalpha():
                                plat = True
                        elif len(license_plate_) == 8 and dua_plat:
                            if license_plate_[0:2].isalpha() and license_plate_[2:6].isdigit() and \
                                    license_plate_[6:].isalpha():
                                plat = True
                        elif len(license_plate_) == 8:
                            if license_plate_[0].isalpha() and license_plate_[1:5].isdigit() and \
                                    license_plate_[5:].isalpha():
                                plat = True
                        elif len(text) == 9:
                            if license_plate_[0:2].isalpha() and license_plate_[2:6].isdigit() and \
                                    license_plate_[6:].isalpha():
                                plat = True

                        # mengecek apakah plat yang terdeteksi sesuai format yang ada
                        if plat:

                            # memasukkan hasil deteksi untuk digunakan oleh frame selanjutnya jika hurufnya tidak
                            # terdeteksi (bisa dibilang menyimpan histori)
                            prev_license_plate_info = {'plat': track_id, 'isi': license_plate_}

                            # menggambar bboxes untuk plat yang berwarna hijau
                            cv2.rectangle(frame, (x_tl, y_tl), (x_tl + int(w), y_tl + int(h)), (0, 255, 0), 4)

                            # membuat sebuah kotak putih diatas plat, yang diisi oleh hasil deteksi huruf pada platnya
                            text_position = (int(x_tl), int(y_tl - (h / 2)))
                            cv2.rectangle(frame, (x_tl, y_tl - int(h)), (x_tl + int(w), y_tl), (255, 255, 255),
                                          cv2.FILLED)
                            cv2.putText(frame, license_plate_, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 0), 4)

                            # menampilkan pada label status isi dari plat yang terdeteksi
                            self.label_status.setStyleSheet("color: rgb(122, 189, 145);")
                            self.label_status.setText(f'plat ke-{track_id} : {license_plate_}')
                        else:
                            # jika id yang saat ini sama dengan id yang ada pada histori prev_license_plate_info
                            if track_id == prev_license_plate_info['plat']:

                                # menggambar bboxes untuk plat yang berwarna hijau
                                cv2.rectangle(frame, (x_tl, y_tl), (x_tl + int(w), y_tl + int(h)), (0, 255, 0), 4)

                                # membuat sebuah kotak putih diatas plat, diisi oleh hasil deteksi huruf pada platnya
                                cv2.rectangle(frame, (x_tl, y_tl - int(h)), (x_tl + int(w), y_tl),
                                              (255, 255, 255), cv2.FILLED)
                                cv2.putText(frame, prev_license_plate_info['isi'], (int(x_tl), int(y_tl - (h / 2))),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)

                                # menampilkan pada label status isi dari plat yang terdeteksi
                                self.label_status.setStyleSheet("color: rgb(122, 189, 145);")
                                self.label_status.setText(f'plat ke-{track_id} : {prev_license_plate_info["isi"]}')
                            else:
                                # menggambar bboxes untuk plat yang berwarna merah
                                cv2.rectangle(frame, (x_tl, y_tl), (x_tl + int(w), y_tl + int(h)), (0, 0, 255), 4)

                                # membuat sebuah kotak putih diatas plat, diisi oleh hasil deteksi huruf pada platnya
                                cv2.rectangle(frame, (x_tl - int(w), y_tl - int(h + 5)), (x_tl + int(w * 2), y_tl),
                                              (255, 255, 255), cv2.FILLED)
                                cv2.putText(frame, 'TIDAK TERDETEKSI', (int(x_tl - int(w)), int(y_tl - (h / 2))),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)

                                # menampilkan pada label status bahwa huruf pada plat tidak terdeteksi
                                self.label_status.setStyleSheet("color: rgb(255, 105, 98);")
                                self.label_status.setText('HURUF PADA PLAT TIDAK TERDETEKSI')

                # jika hasil deteksi plat tidak memiliki ID, maka tidak akan dianggap plat
                else:

                    # memberi feedback tidak ada plat pada label status dan mengosongkan semua label prose dan hasil
                    self.label_status.setStyleSheet("color: rgb(255, 105, 98);")
                    self.label_status.setText(f'TIDAK ADA PLAT')
                    self.label_p2.setPixmap(QPixmap())
                    self.label_p3.setPixmap(QPixmap())
                    self.label_p4.setPixmap(QPixmap())
                    self.label_p5.setPixmap(QPixmap())
                    self.label_p6.setPixmap(QPixmap())

                frame_count += 1

                # menampilkan frame apa saat ini yang diproses dari jumlah frame total
                self.label_frame.setText(f'Frame {frame_count}/{total_frames-1}')

                # menambahkan frame yang diproses ke video yang akan disimpan
                out.write(frame)

                # menampilkan frame hasil olahan yang telah diberikan rectangle dan text ke label hasil
                self.Image = frame
                self.displayImage(self.label_hasil)

                # memberikan delay
                cv2.waitKey(1)

            # jika video sudah beres, maka akan keluar dari loop
            else:
                break

        # menutup video yang sudah selesai diproses
        cap.release()

    # fungsi untuk mengubah sebuah numpy array menjadi pixmap yang akan ditampilkan di label
    def displayImage(self, label):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image,
                     self.Image.shape[1],
                     self.Image.shape[0],
                     self.Image.strides[0], qformat)
        img = img.rgbSwapped()

        label.setScaledContents(True)
        label.setPixmap(QPixmap.fromImage(img))
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


# Pengaturan aplikasi Qt dan inisialisasi jendela ShowImage
# Serta menampilkan jendela dengan judul yang ditentukan
app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Tugas Akhir')
window.show()
window.start()
sys.exit(app.exec_())

