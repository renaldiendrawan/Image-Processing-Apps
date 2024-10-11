import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('arithmetical.ui', self)

        # Find and connect actions
        self.actionGambar_1 = self.findChild(QtWidgets.QAction, 'actionGambar_1')
        self.actionGambar_2 = self.findChild(QtWidgets.QAction, 'actionGambar_2')
        self.actionPenjumlahan = self.findChild(QtWidgets.QAction, 'actionPenjumlahan')
        self.actionPengurangan = self.findChild(QtWidgets.QAction, 'actionPengurangan')
        self.actionPerkalian = self.findChild(QtWidgets.QAction, 'actionPerkalian')
        self.actionPembagian = self.findChild(QtWidgets.QAction, 'actionPembagian')
        self.actionAND = self.findChild(QtWidgets.QAction, 'actionAND')
        self.actionOR = self.findChild(QtWidgets.QAction, 'actionOR')
        self.actionXOR = self.findChild(QtWidgets.QAction, 'actionXOR')
        self.actionKeluar = self.findChild(QtWidgets.QAction, 'actionKeluar')
        self.actionTentang_Aplikasi = self.findChild(QtWidgets.QAction, 'actionTentang_Aplikasi')
        
        self.actionClearOutput = self.findChild(QtWidgets.QAction, 'actionClearOutput')
        self.actionClearAll = self.findChild(QtWidgets.QAction, 'actionClearAll')

        # Connect actions
        self.actionKeluar.triggered.connect(self.close)
        self.actionTentang_Aplikasi.triggered.connect(self.show_about_dialog)
        self.actionGambar_1.triggered.connect(self.open_image_1)
        self.actionGambar_2.triggered.connect(self.open_image_2)
        self.actionPenjumlahan.triggered.connect(self.image_addition)
        self.actionPengurangan.triggered.connect(self.image_subtraction)
        self.actionPerkalian.triggered.connect(self.image_multiplication)
        self.actionPembagian.triggered.connect(self.image_division)
        self.actionAND.triggered.connect(self.image_and)
        self.actionOR.triggered.connect(self.image_or)
        self.actionXOR.triggered.connect(self.image_xor)
        
        self.actionClearOutput.triggered.connect(self.clear_output)
        self.actionClearAll.triggered.connect(self.clear_all)

        # Variables to hold images
        self.image1 = None
        self.image2 = None

    def show_about_dialog(self):
        QMessageBox.about(self, "Tentang Aplikasi", "Ini adalah aplikasi aritmatika sederhana.")

    def open_image_1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 1", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image1 = cv2.imread(file_name)
            self.display_image(self.image1, self.graphicsView)

    def open_image_2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 2", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name)
            self.display_image(self.image2, self.graphicsView_3)

    def display_image(self, image, graphics_view):
        if image is not None:
            # Convert BGR (OpenCV format) to RGB (Qt format)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimg = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            graphics_view.setScene(scene)
            graphics_view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)


    def resize_images(self):
        """Resize the second image to match the first one."""
        if self.image1 is not None and self.image2 is not None:
            height, width = self.image1.shape[:2]
            self.image2 = cv2.resize(self.image2, (width, height))

    def match_channels(self):
        """Convert images to have the same number of channels."""
        if self.image1 is not None and self.image2 is not None:
            if len(self.image1.shape) != len(self.image2.shape):
                if len(self.image1.shape) == 2:
                    self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
                elif len(self.image2.shape) == 2:
                    self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)

    def check_images_loaded(self):
        """Ensure both images are loaded before performing operations."""
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Error", "Both images must be loaded.")
            return False
        return True

    # Arithmetic operation functions
    def show_result(self, result):
        if result is not None:
            self.display_image(result, self.graphicsView_2)

    def image_addition(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.add(self.image1, self.image2)
            self.show_result(result)

    def image_subtraction(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.subtract(self.image1, self.image2)
            self.show_result(result)

    def image_multiplication(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.multiply(self.image1, self.image2)
            self.show_result(result)

    def image_division(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.divide(self.image1, self.image2)
            self.show_result(result)

    def image_and(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.bitwise_and(self.image1, self.image2)
            self.show_result(result)

    def image_or(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.bitwise_or(self.image1, self.image2)
            self.show_result(result)

    def image_xor(self):
        if self.check_images_loaded():
            self.resize_images()
            self.match_channels()
            result = cv2.bitwise_xor(self.image1, self.image2)
            self.show_result(result)
            
    def clear_output(self):
        # Menghapus isi dari graphicsView_2 saja
        self.graphicsView_2.scene().clear()
        
    def clear_all(self):
        # Menghapus isi dari semua graphicsView: graphicsView, graphicsView_2, dan graphicsView_3
        self.graphicsView.scene().clear()
        self.graphicsView_2.scene().clear()
        self.graphicsView_3.scene().clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
