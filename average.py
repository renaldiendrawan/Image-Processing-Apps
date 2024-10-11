from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import skimage.feature as skft
import pandas as pd
import os
from arithmetical import MyWindow as ArithmeticalWindow

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        # Load UI from .ui file
        uic.loadUi("ProjectPythonUts.ui", self)

        # Find widgets
        self.graphicsView = self.findChild(QtWidgets.QGraphicsView, 'graphicsView')
        self.graphicsView_2 = self.findChild(QtWidgets.QGraphicsView, 'graphicsView_2')
        self.actionBuka = self.findChild(QtWidgets.QAction, 'actionBuka')
        self.actionSimpan_Sebagai = self.findChild(QtWidgets.QAction, 'actionSimpan_Sebagai')
        self.actionKeluar = self.findChild(QtWidgets.QAction, 'actionKeluar')
        
        self.actionAverage = self.findChild(QtWidgets.QAction, 'actionAverage')
        self.actionLightness = self.findChild(QtWidgets.QAction, 'actionLightness')
        self.actionLuminance = self.findChild(QtWidgets.QAction, 'actionLuminance')
        self.actionQuantization = self.findChild(QtWidgets.QAction, 'actionQuantization')
        self.actionContrast = self.findChild(QtWidgets.QAction, 'actionContrast')
        self.actionBrightness = self.findChild(QtWidgets.QAction, 'actionBrightness')
        self.actionSaturation = self.findChild(QtWidgets.QAction, 'actionSaturation')
        self.actionNegation = self.findChild(QtWidgets.QAction, 'actionGamma_Corection')
        self.actionLogBrightness = self.findChild(QtWidgets.QAction, 'actionLog_Brightness')
        self.actionKuning = self.findChild(QtWidgets.QAction, 'actionKuning')
        self.actionOrange = self.findChild(QtWidgets.QAction, 'actionOrange')
        self.actionCyan = self.findChild(QtWidgets.QAction, 'actionCyan')
        self.actionPurple = self.findChild(QtWidgets.QAction, 'actionPurple')
        self.actionGrey = self.findChild(QtWidgets.QAction, 'actionGrey')
        self.actionCoklat = self.findChild(QtWidgets.QAction, 'actionCoklat')
        self.actionMerah = self.findChild(QtWidgets.QAction, 'actionMerah')
        self.actionInvert = self.findChild(QtWidgets.QAction, 'actionInvert')
        self.actionHistogram_Qualization = self.findChild(QtWidgets.QAction, 'actionHistogram_Qualization')
        self.actionFuzzy_HE_RGB = self.findChild(QtWidgets.QAction, 'actionFuzzy_HE_RGB')
        self.actionFuzzy_Grayscale = self.findChild(QtWidgets.QAction, 'actionFuzzy_Grayscale')
        self.actionTentang_Aplikasi = self.findChild(QtWidgets.QAction, 'actionTentang_Aplikasi')
        self.actionTranslate_Image = self.findChild(QtWidgets.QAction, 'actionTranslate_Image')
        self.actionRotate_Image = self.findChild(QtWidgets.QAction, 'actionRotate_Image')
        self.actionFlip_Image = self.findChild(QtWidgets.QAction, 'actionFlip_Image')
        self.actionZoom_Image = self.findChild(QtWidgets.QAction, 'actionZoom_Image')
        self.actionCrop_Image = self.findChild(QtWidgets.QAction, 'actionCrop_Image')
        
        self.actionIdentity = self.findChild(QtWidgets.QAction, 'actionIdentity')
        self.actionSobel = self.findChild(QtWidgets.QAction, 'actionSobel')
        self.actionPrewitt = self.findChild(QtWidgets.QAction, 'actionPrewitt')
        self.actionCanny = self.findChild(QtWidgets.QAction, 'actionCanny')
        self.actionSharpen = self.findChild(QtWidgets.QAction, 'actionSharpen')
        self.actionGaussian_Blur_3x3 = self.findChild(QtWidgets.QAction, 'actionGaussian_Blur_3x3')
        self.actionGaussian_Blur_5x5 = self.findChild(QtWidgets.QAction, 'actionGaussian_Blur_5x5')
        self.actionUnsharp_Masking = self.findChild(QtWidgets.QAction, 'actionUnsharp_Masking')
        self.actionAverage_Filter = self.findChild(QtWidgets.QAction, 'actionAverage_Filter')
        self.actionLow_Pass_Filter = self.findChild(QtWidgets.QAction, 'actionLow_Pass_Filter')
        self.actionHeigh_Pass_Filter = self.findChild(QtWidgets.QAction, 'actionHeigh_Pass_Filter')
        self.actionBandstop_Filter = self.findChild(QtWidgets.QAction, 'actionBandstop_Filter')
        
        self.actionErosion = self.findChild(QtWidgets.QAction, 'actionErosion')
        self.actionDilation = self.findChild(QtWidgets.QAction, 'actionDilation')
        self.actionOpening = self.findChild(QtWidgets.QAction, 'actionOpening')
        self.actionClosing = self.findChild(QtWidgets.QAction, 'actionClosing')
        self.actionHitormiss = self.findChild(QtWidgets.QAction, 'actionHitormiss')
        self.actionThinning = self.findChild(QtWidgets.QAction, 'actionThinning')
        self.actionSkeletonization = self.findChild(QtWidgets.QAction, 'actionSkeletonization')
        self.actionPruning = self.findChild(QtWidgets.QAction, 'actionPruning')
        
        self.actionRegion_growing = self.findChild(QtWidgets.QAction, 'actionRegion_growing')
        self.actionKMeans_clustering = self.findChild(QtWidgets.QAction, 'actionKMeans_clustering')
        self.actionWatershed_segmentation = self.findChild(QtWidgets.QAction, 'actionWatershed_segmentation')
        self.actionGlobal_thresholding = self.findChild(QtWidgets.QAction, 'actionGlobal_thresholding')
        self.actionAdaptive_thresholding = self.findChild(QtWidgets.QAction, 'actionAdaptive_thresholding')
        
        self.actionManual = self.findChild(QtWidgets.QAction, 'actionManual')
        self.actionOpenCV = self.findChild(QtWidgets.QAction, 'actionOpenCV')
        
        self.actionCitraWarnaRGB = self.findChild(QtWidgets.QAction, 'actionCitraWarnaRGB')
        self.actionTeksturdenganGLCM = self.findChild(QtWidgets.QAction, 'actionTeksturdenganGLCM')
        
        self.actionClearOutput = self.findChild(QtWidgets.QAction, 'actionClearOutput')
        self.actionClearAll = self.findChild(QtWidgets.QAction, 'actionClearAll')

           # Add actions for histogram
        self.actionInputHistogram = self.findChild(QtWidgets.QAction, 'actionInput')
        self.actionOutputHistogram = self.findChild(QtWidgets.QAction, 'actionOutput')
        self.actionInputOutputHistogram = self.findChild(QtWidgets.QAction, 'actionInput_Output')
        self.actionArithmetical_Operation = self.findChild(QtWidgets.QAction, 'actionArithmetical_Operation')

        # Find bit depth actions
        self.action1_Bit = self.findChild(QtWidgets.QAction, 'action1_Bit')
        self.action2_Bit = self.findChild(QtWidgets.QAction, 'action2_Bit')
        self.action3_Bit = self.findChild(QtWidgets.QAction, 'action3_Bit')
        self.action4_Bit = self.findChild(QtWidgets.QAction, 'action4_Bit')
        self.action5_Bit = self.findChild(QtWidgets.QAction, 'action5_Bit')
        self.action6_Bit = self.findChild(QtWidgets.QAction, 'action6_Bit')
        self.action7_Bit = self.findChild(QtWidgets.QAction, 'action7_Bit')

        # Connect actions to functions
        self.actionBuka.triggered.connect(self.buka_gambar)
        self.actionSimpan_Sebagai.triggered.connect(self.simpan_sebagai)
        self.actionKeluar.triggered.connect(self.keluar_aplikasi)
        
        self.actionAverage.triggered.connect(self.convert_to_grayscale_average)
        self.actionLightness.triggered.connect(self.convert_to_grayscale_lightness)
        self.actionLuminance.triggered.connect(self.convert_to_grayscale_luminance)
        self.actionQuantization.triggered.connect(self.perform_quantization)
        self.actionContrast.triggered.connect(self.adjust_contrast)
        self.actionBrightness.triggered.connect(self.adjust_brightness)
        self.actionSaturation.triggered.connect(self.adjust_saturation)
        self.actionNegation.triggered.connect(self.perform_negation)
        self.actionLogBrightness.triggered.connect(self.perform_log_brightness)
        self.actionKuning.triggered.connect(self.convert_to_yellow)
        self.actionOrange.triggered.connect(self.convert_to_orange)
        self.actionCyan.triggered.connect(self.convert_to_cyan)
        self.actionPurple.triggered.connect(self.convert_to_purple)
        self.actionGrey.triggered.connect(self.convert_to_grey)
        self.actionCoklat.triggered.connect(self.convert_to_brown)
        self.actionMerah.triggered.connect(self.convert_to_red)
        self.actionInvert.triggered.connect(self.perform_invert)
        self.actionHistogram_Qualization.triggered.connect(self.histogram_equalization)
        self.actionFuzzy_HE_RGB.triggered.connect(self.fuzzy_histogram_equalization)
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzy_grayscale)
        self.actionTentang_Aplikasi.triggered.connect(self.show_tentang_dialog)
        self.actionTranslate_Image.triggered.connect(self.translate_image)
        self.actionRotate_Image.triggered.connect(self.rotate_image)
        self.actionFlip_Image.triggered.connect(self.flip_image)
        self.actionZoom_Image.triggered.connect(self.zoom_image)
        self.actionCrop_Image.triggered.connect(self.crop_image)
        
        self.actionIdentity.triggered.connect(self.identity)
        self.actionSobel.triggered.connect(self.sobel)
        self.actionPrewitt.triggered.connect(self.prewitt)
        self.actionCanny.triggered.connect(self.canny)
        self.actionSharpen.triggered.connect(self.sharpen)
        self.actionGaussian_Blur_3x3.triggered.connect(self.gaussian_blur_3x3)
        self.actionGaussian_Blur_5x5.triggered.connect(self.gaussian_blur_5x5)
        self.actionUnsharp_Masking.triggered.connect(self.unsharp_masking)
        self.actionAverage_Filter.triggered.connect(self.average_filter)
        self.actionLow_Pass_Filter.triggered.connect(self.low_pass_filter)
        self.actionHeigh_Pass_Filter.triggered.connect(self.heigh_pass_filter)
        self.actionBandstop_Filter.triggered.connect(self.bandstop_filter)
        
        self.actionErosion.triggered.connect(self.erosion)
        self.actionDilation.triggered.connect(self.dilation)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)
        self.actionHitormiss.triggered.connect(self.hitormiss)
        self.actionThinning.triggered.connect(self.thinning)
        self.actionSkeletonization.triggered.connect(self.skeletonization)
        self.actionPruning.triggered.connect(self.pruning)
        
        self.actionRegion_growing.triggered.connect(self.region_growing)
        self.actionKMeans_clustering.triggered.connect(self.kmeans_clustering)
        self.actionWatershed_segmentation.triggered.connect(self.watershed_segmentation)
        self.actionGlobal_thresholding.triggered.connect(self.global_thresholding)
        self.actionAdaptive_thresholding.triggered.connect(self.adaptive_thresholding)
        
        self.actionManual.triggered.connect(self.manual_convolution)
        self.actionOpenCV.triggered.connect(self.opencv_convolution)
        
        self.actionCitraWarnaRGB.triggered.connect(self.citra_warna_rgb)
        self.actionTeksturdenganGLCM.triggered.connect(self.tekstur_dengan_glcm)
        
        self.actionClearOutput.triggered.connect(self.clear_output)
        self.actionClearAll.triggered.connect(self.clear_all)

         # Connect histogram actions to functions
        self.actionInputHistogram.triggered.connect(self.show_input_histogram)
        self.actionOutputHistogram.triggered.connect(self.show_output_histogram)
        self.actionInputOutputHistogram.triggered.connect(self.show_input_output_histogram)
        self.actionArithmetical_Operation.triggered.connect(self.show_arithmetical_dialog)

        # Connect bit depth actions to function
        self.action1_Bit.triggered.connect(lambda: self.set_bit_depth(1))
        self.action2_Bit.triggered.connect(lambda: self.set_bit_depth(2))
        self.action3_Bit.triggered.connect(lambda: self.set_bit_depth(3))
        self.action4_Bit.triggered.connect(lambda: self.set_bit_depth(4))
        self.action5_Bit.triggered.connect(lambda: self.set_bit_depth(5))
        self.action6_Bit.triggered.connect(lambda: self.set_bit_depth(6))
        self.action7_Bit.triggered.connect(lambda: self.set_bit_depth(7))

        self.image = None
        self.bit_depth = 4  # Default bit depth

    def buka_gambar(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Image Files (*.png *.jpg *.bmp)")
        
        if file_name:
            # Load the image into a QPixmap
            pixmap = QtGui.QPixmap(file_name)

            # Create a QGraphicsScene
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

            # Load the image into a numpy array
            self.image = cv2.imread(file_name)
            self.processed_image = None 

    def simpan_sebagai(self):
        if self.graphicsView_2.scene():
            pixmap = self.graphicsView_2.scene().items()[0].pixmap()
            file_name, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar Sebagai", "", "Image Files (*.png *.jpg *.bmp)")
            if file_name:
                pixmap.save(file_name)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Tidak ada gambar untuk disimpan.")

    def keluar_aplikasi(self):
        QtWidgets.QApplication.quit()

    def show_input_histogram(self):
        if self.image is not None:
            plt.figure(figsize=(10, 5))
            if len(self.image.shape) == 2:  # Grayscale image
                plt.hist(self.image.ravel(), 256, [0, 256], color='black')
                plt.title("Input Image Histogram (Grayscale)")
            else:  # RGB image
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    plt.hist(self.image[..., i].ravel(), 256, [0, 256], color=color)
                plt.title("Input Image Histogram (RGB)")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def show_output_histogram(self):
        if self.processed_image is not None:
            plt.figure(figsize=(10, 5))
            if len(self.processed_image.shape) == 2:  # Grayscale image
                plt.hist(self.processed_image.ravel(), 256, [0, 256], color='black')
                plt.title("Output Image Histogram (Grayscale)")
            else:  # RGB image
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    plt.hist(self.processed_image[..., i].ravel(), 256, [0, 256], color=color)
                plt.title("Output Image Histogram (RGB)")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please process the image first.")

    def show_input_output_histogram(self):
        if self.image is not None and self.processed_image is not None:
            plt.figure(figsize=(10, 5))

            # Input Image Histogram
            plt.subplot(1, 2, 1)
            if len(self.image.shape) == 2:  # Grayscale image
                plt.hist(self.image.ravel(), 256, [0, 256], color='black')
                plt.title("Input Image Histogram (Grayscale)")
            else:  # RGB image
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    plt.hist(self.image[..., i].ravel(), 256, [0, 256], color=color)
                plt.title("Input Image Histogram (RGB)")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")

            # Output Image Histogram
            plt.subplot(1, 2, 2)
            if len(self.processed_image.shape) == 2:  # Grayscale image
                plt.hist(self.processed_image.ravel(), 256, [0, 256], color='black')
                plt.title("Output Image Histogram (Grayscale)")
            else:  # RGB image
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    plt.hist(self.processed_image[..., i].ravel(), 256, [0, 256], color=color)
                plt.title("Output Image Histogram (RGB)")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open and process an image first.")

    # Helper function to plot histogram using matplotlib
    def plot_histogram(self, img, title="Histogram", subplot=None):
        # Convert image to grayscale if it's in color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # Plot histogram
        if subplot:
            plt.subplot(1, 2, subplot)
            plt.title(title)
        else:
            plt.figure()
            plt.title(title)
        
        plt.plot(hist, color='black')
        plt.xlim([0, 256])
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        if not subplot:
            plt.show()

    # Fungsi Konversi Warna RGB
    def convert_to_yellow(self):
        if self.image is not None:
            yellow_img = self.image.copy()
            yellow_img[:, :, 0] = 0  # Set B channel to 0
            self.processed_image = yellow_img
            self.display_image_in_view(yellow_img)

    def convert_to_orange(self):
        if self.image is not None:
            orange_img = self.image.copy()
            orange_img[:, :, 1] = np.minimum(orange_img[:, :, 1], 128)  # Set G channel to lower value
            orange_img[:, :, 0] = 0  # Set B channel to 0
            self.processed_image = orange_img
            self.display_image_in_view(orange_img)

    def convert_to_cyan(self):
        if self.image is not None:
            cyan_img = self.image.copy()
            cyan_img[:, :, 2] = 0  # Set R channel to 0
            self.processed_image = cyan_img
            self.display_image_in_view(cyan_img)

    def convert_to_purple(self):
        if self.image is not None:
            purple_img = self.image.copy()
            purple_img[:, :, 1] = 0  # Set G channel to 0
            self.processed_image = purple_img
            self.display_image_in_view(purple_img)

    def convert_to_grey(self):
        if self.image is not None:
            grey_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            grey_img = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB)  # Convert back to 3-channel image for display
            self.processed_image = grey_img
            self.display_image_in_view(grey_img)

    def convert_to_brown(self):
        if self.image is not None:
            brown_img = self.image.copy()
            brown_img[:, :, 0] = np.minimum(brown_img[:, :, 0], 100)  # Reduce B channel
            brown_img[:, :, 1] = np.minimum(brown_img[:, :, 1], 50)  # Reduce G channel
            self.processed_image = brown_img
            self.display_image_in_view(brown_img)

    def convert_to_red(self):
        if self.image is not None:
            red_img = self.image.copy()
            red_img[:, :, 1] = 0  # Set G channel to 0
            red_img[:, :, 0] = 0  # Set B channel to 0
            self.processed_image = red_img
            self.display_image_in_view(red_img)

    def convert_to_grayscale_average(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = gray_image
            self.display_image_in_view(gray_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def convert_to_grayscale_lightness(self):
        if self.image is not None:
            lightness_image = np.mean(self.image, axis=2).astype(np.uint8)
            self.processed_image = lightness_image
            self.display_image_in_view(lightness_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def convert_to_grayscale_luminance(self):
        if self.image is not None:
            luminance_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = luminance_image
            self.display_image_in_view(luminance_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def perform_quantization(self):
        if self.image is not None:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Quantization parameters
            levels = 2 ** self.bit_depth
            interval_size = 256 // levels
            intervals = [i * interval_size for i in range(levels)]
            mid_values = [((i * interval_size) + ((i + 1) * interval_size - 1)) // 2 for i in range(levels)]
            
            # Quantize function
            def quantize(image, intervals, mid_values):
                quantized_image = np.zeros_like(image)
                for i in range(len(intervals)):
                    lower_bound = intervals[i]
                    upper_bound = lower_bound + interval_size - 1
                    mask = (image >= lower_bound) & (image <= upper_bound)
                    quantized_image[mask] = mid_values[i]
                return quantized_image
            
            # Apply quantization
            quantized_image = quantize(image_gray, intervals, mid_values)
            self.processed_image = quantized_image
            self.display_image_in_view(quantized_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def adjust_contrast(self):
        if self.image is not None:
            alpha = 1.5  # Contrast control (1.0-3.0)
            adjusted_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
            self.processed_image = adjusted_image
            self.display_image_in_view(adjusted_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def adjust_brightness(self):
        if self.image is not None:
            beta = 50  # Brightness control (0-100)
            adjusted_image = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)
            self.processed_image = adjusted_image
            self.display_image_in_view(adjusted_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def adjust_saturation(self):
        if self.image is not None:
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_image[..., 1] = hsv_image[..., 1] * 1.5  # Saturation control (1.0-3.0)
            adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            self.processed_image = adjusted_image
            self.display_image_in_view(adjusted_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def perform_negation(self):
        if self.image is not None:
            negated_image = cv2.bitwise_not(self.image)
            self.processed_image = negated_image
            self.display_image_in_view(negated_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def perform_log_brightness(self):
        if self.image is not None:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            c = 255 / (np.log(1 + np.max(image_gray)))
            log_brightness_image = c * np.log(1 + image_gray)
            log_brightness_image = np.array(log_brightness_image, dtype=np.uint8)
            self.processed_image = log_brightness_image
            self.display_image_in_view(log_brightness_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def perform_invert(self):
        if self.image is not None:
            inverted_image = 255 - self.image
            self.processed_image = inverted_image
            self.display_image_in_view(inverted_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def histogram_equalization(self):
        if self.image is not None:
            if len(self.image.shape) == 3:  # RGB image
                ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                equalized_y = cv2.equalizeHist(y)
                ycrcb = cv2.merge([equalized_y, cr, cb])
                self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:  # Grayscale image
                self.processed_image = cv2.equalizeHist(self.image)
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def fuzzy_histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Hitung mean dan standar deviasi
            mean = np.mean(gray_image)
            stddev = np.std(gray_image)

            # Hitung nilai keanggotaan fuzzy
            membership = self.fuzzy_membership_function(gray_image, mean, stddev)

            # Normalisasi ke rentang 0-255
            fuzzy_image = 255 * (membership / np.max(membership))
            fuzzy_image = fuzzy_image.astype(np.uint8)

            # Tampilkan gambar hasil
            self.display_image_in_view(fuzzy_image)
        else:
            print("Gambar belum dibuka.")

    def fuzzy_membership_function(self, image, mean, stddev):
        # Fungsi ini menghitung nilai keanggotaan fuzzy berdasarkan mean dan stddev
        membership = np.exp(-0.5 * ((image - mean) / stddev) ** 2)
        return membership

    def fuzzy_grayscale(self):
        # Fuzzy Histogram Equalization for Grayscale images (basic implementation)
        if self.image is not None:
            # Apply a basic fuzzy HE technique for grayscale images (placeholder)
            self.processed_image = self.image  # Dummy example
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    # 1. Translation (Translasi)
    def translate_image(self):
        if self.image is not None:
            # Input manual untuk translasi
            x_translation, ok1 = QtWidgets.QInputDialog.getInt(self, "Input X Translation", "Enter the X translation value (in pixels):", 50)
            y_translation, ok2 = QtWidgets.QInputDialog.getInt(self, "Input Y Translation", "Enter the Y translation value (in pixels):", 50)

            if ok1 and ok2:
                # Define translation matrix
                rows, cols, _ = self.image.shape
                M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])  # Translasi dengan nilai input
                self.processed_image = cv2.warpAffine(self.image, M, (cols, rows))
                self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    # 2. Rotation (Rotasi)
    def rotate_image(self):
        if self.image is not None:
            # Input manual untuk sudut rotasi
            angle, ok = QtWidgets.QInputDialog.getDouble(self, "Input Rotation Angle", "Enter the rotation angle (in degrees):", 45)

            if ok:
                # Define rotation matrix
                rows, cols, _ = self.image.shape
                center = (cols // 2, rows // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)  # Rotasi dengan sudut yang diinput
                self.processed_image = cv2.warpAffine(self.image, M, (cols, rows))
                self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    # 3. Flipping (Refleksi)
    def flip_image(self):
        if self.image is not None:
            # Input manual untuk arah flipping
            flip_direction, ok = QtWidgets.QInputDialog.getInt(self, "Input Flip Direction", "Enter 1 for horizontal flip, 0 for vertical flip:", 1)

            if ok:
                # Flip image berdasarkan input
                self.processed_image = cv2.flip(self.image, flip_direction)
                self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    # 4. Zooming (Skala)
    def zoom_image(self):
        if self.image is not None:
            # Input manual untuk faktor zoom
            zoom_factor, ok = QtWidgets.QInputDialog.getDouble(self, "Input Zoom Factor", "Enter the zoom factor (e.g., 1.5 for 150%):", 1.5)

            if ok:
                # Define zoom matrix
                rows, cols, _ = self.image.shape
                center = (cols // 2, rows // 2)
                M = cv2.getRotationMatrix2D(center, 0, zoom_factor)  # Zoom dengan faktor yang diinput
                self.processed_image = cv2.warpAffine(self.image, M, (cols, rows))
                self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    # 5. Cropping dengan ruang hitam di sekitar hasil cropping
    def crop_image(self):
        if self.image is not None:
            # Cek dimensi gambar asli
            rows, cols, _ = self.image.shape

            # Input manual untuk koordinat cropping
            left, ok1 = QtWidgets.QInputDialog.getInt(self, "Input Left Coordinate", f"Enter the left coordinate (0 - {cols}):", 50, 0, cols)
            top, ok2 = QtWidgets.QInputDialog.getInt(self, "Input Top Coordinate", f"Enter the top coordinate (0 - {rows}):", 50, 0, rows)
            right, ok3 = QtWidgets.QInputDialog.getInt(self, "Input Right Coordinate", f"Enter the right coordinate ({left} - {cols}):", 350, left, cols)
            bottom, ok4 = QtWidgets.QInputDialog.getInt(self, "Input Bottom Coordinate", f"Enter the bottom coordinate ({top} - {rows}):", 250, top, rows)

            if ok1 and ok2 and ok3 and ok4:
                # Validasi koordinat agar berada dalam batas gambar
                left = max(0, min(left, cols))
                top = max(0, min(top, rows))
                right = max(left, min(right, cols))
                bottom = max(top, min(bottom, rows))

                # Crop gambar
                cropped_image = self.image[top:bottom, left:right]

                # Ukuran hasil cropping
                crop_height, crop_width, _ = cropped_image.shape

                # Buat gambar hitam dengan ukuran sama seperti gambar asli
                black_background = np.zeros((rows, cols, 3), dtype=np.uint8)  # Gambar hitam dengan dimensi gambar asli

                # Tempatkan gambar yang dicrop di posisi yang sesuai pada background hitam
                # Kita menggunakan bagian dari latar hitam yang ukurannya sesuai dengan hasil cropping
                black_background[top:bottom, left:right] = cropped_image

                # Tampilkan gambar hasil cropping dengan latar hitam
                self.processed_image = black_background
                self.display_image_in_view(self.processed_image)

                # Update gambar asli dengan hasil yang dicrop dan diberi latar hitam
                self.image = self.processed_image
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Cropping operation was canceled.")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
          
    def sobel(self):
        if self.image is not None:
            try:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                # Sobel Edge Detection
                sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = cv2.magnitude(sobel_x, sobel_y)

                # Tampilkan hasil deteksi tepi Sobel
                self.display_image_in_view(sobel_combined)

                # Simpan hasil ke self.processed_image
                self.processed_image = sobel_combined
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred during Sobel edge detection.")
                print(f"Error in apply_sobel_edge_detection: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def prewitt(self):
        if self.image is not None:
            try:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                # Prewitt Edge Detection (Manual)
                prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
                prewitt_y = cv2.filter2D(gray_image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
                prewitt_combined = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))

                # Tampilkan hasil deteksi tepi Prewitt
                self.display_image_in_view(prewitt_combined)

                # Simpan hasil ke self.processed_image
                self.processed_image = prewitt_combined
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred during Prewitt edge detection.")
                print(f"Error in apply_prewitt_edge_detection: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def canny(self):
        if self.image is not None:
            try:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                # Canny Edge Detection
                canny_edges = cv2.Canny(gray_image, 100, 200)

                # Tampilkan hasil deteksi tepi Canny
                self.display_image_in_view(canny_edges)

                # Simpan hasil ke self.processed_image
                self.processed_image = canny_edges
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred during Canny edge detection.")
                print(f"Error in apply_canny_edge_detection: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
            
    def apply_filter(self, kernel):
        if self.image is not None:
            try:
                # Terapkan filter dengan konvolusi
                filtered_image = cv2.filter2D(self.image, -1, kernel)
                
                # Tampilkan hasil filter
                self.display_image_in_view(filtered_image)

                # Simpan hasil ke self.processed_image
                self.processed_image = filtered_image
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred while applying the filter.")
                print(f"Error in apply_filter: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def identity(self):
        identity_kernel = np.array([[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]])
        self.apply_filter(identity_kernel)

    def sharpen(self):
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        self.apply_filter(sharpen_kernel)

    def gaussian_blur_3x3(self):
        gaussian_blur_3x3_kernel = cv2.getGaussianKernel(3, 0)
        gaussian_blur_3x3 = np.outer(gaussian_blur_3x3_kernel, gaussian_blur_3x3_kernel)
        self.apply_filter(gaussian_blur_3x3)

    def gaussian_blur_5x5(self):
        gaussian_blur_5x5_kernel = cv2.getGaussianKernel(5, 0)
        gaussian_blur_5x5 = np.outer(gaussian_blur_5x5_kernel, gaussian_blur_5x5_kernel)
        self.apply_filter(gaussian_blur_5x5)

    def unsharp_masking(self):
        if self.image is not None:
            try:
                # Convert to grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                # Gaussian Blur
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)
                
                # Unsharp Masking
                unsharp_image = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)
                
                # Display the result
                self.display_image_in_view(unsharp_image)

                # Save result to self.processed_image
                self.processed_image = unsharp_image
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred during unsharp masking.")
                print(f"Error in unsharp_masking: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")

    def average_filter(self):
        average_kernel = np.ones((3, 3), np.float32) / 9
        self.apply_filter(average_kernel)

    def low_pass_filter(self):
        low_pass_kernel = np.ones((5, 5), np.float32) / 25
        self.apply_filter(low_pass_kernel)

    def heigh_pass_filter(self):
        high_pass_kernel = np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]])
        self.apply_filter(high_pass_kernel)

    def bandstop_filter(self):
        if self.image is not None:
            try:
                # Konversi gambar ke grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                # Transformasi Fourier
                dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                rows, cols = gray_image.shape
                crow, ccol = rows // 2, cols // 2

                # Membuat masker bandstop
                mask = np.ones((rows, cols, 2), np.uint8)
                r_out = 60  # Radius luar untuk stop band
                r_in = 30  # Radius dalam untuk stop band
                cv2.circle(mask, (ccol, crow), r_out, 0, thickness=-1)
                cv2.circle(mask, (ccol, crow), r_in, 1, thickness=-1)

                # Terapkan masker ke hasil DFT
                fshift = dft_shift * mask

                # Transformasi balik (Inverse FFT)
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

                # Normalisasi gambar hasil
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
                img_back = np.uint8(img_back)

                # Tampilkan hasil
                self.display_image_in_view(img_back)

                # Simpan hasil ke self.processed_image
                self.processed_image = img_back
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Warning", "An error occurred during bandstop filtering.")
                print(f"Error in bandstop_filter: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open an image first.")
    
    def region_growing(self):
        seed_point = (50, 50)  # Atur nilai seed_point yang dipilih
        threshold = 30  # Atur nilai threshold yang diinginkan
        self.region_growing_algorithm(seed_point, threshold)

    def region_growing_algorithm(self, seed_point, threshold=10):
        if self.image is not None:
            # Meminta user untuk memilih seed point secara manual (misalnya melalui GUI)
            seed_x, ok_x = QtWidgets.QInputDialog.getInt(self, "Input Seed Point X", "Enter X coordinate:")
            seed_y, ok_y = QtWidgets.QInputDialog.getInt(self, "Input Seed Point Y", "Enter Y coordinate:")

            if not ok_x or not ok_y:
                QtWidgets.QMessageBox.warning(self, "Warning", "Seed point input was canceled.")
                return

            seed_point = (seed_x, seed_y)

            # Konversi gambar ke grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Validasi seed_point
            if not isinstance(seed_point, tuple) or len(seed_point) != 2:
                print("Error: seed_point is not valid. It must be a tuple of (x, y).")
                return  # Hentikan eksekusi jika seed_point tidak valid

            x_seed, y_seed = seed_point

            # Cek apakah seed_point berada dalam batas gambar
            if x_seed < 0 or y_seed < 0 or x_seed >= gray_image.shape[1] or y_seed >= gray_image.shape[0]:
                print("Error: seed_point is out of image bounds.")
                return

            # Inisialisasi gambar biner untuk area hasil region growing
            region_grown = np.zeros_like(gray_image)

            # Stack untuk menyimpan pixel yang akan diperiksa
            stack = [seed_point]

            while stack:
                current_point = stack.pop()
                x, y = current_point

                # Cek apakah pixel berada dalam batas gambar
                if x < 0 or y < 0 or x >= gray_image.shape[1] or y >= gray_image.shape[0]:
                    continue

                if region_grown[y, x] == 0 and abs(int(gray_image[y, x]) - int(gray_image[y_seed, x_seed])) < threshold:
                    region_grown[y, x] = 255  # Tandai sebagai bagian dari region
                    # Tambahkan piksel tetangga ke stack
                    stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

            self.processed_image = region_grown
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or crop an image first.")

    def kmeans_clustering(self):
        if self.image is not None:
            # Minta input manual dari pengguna untuk nilai k
            k, ok = QtWidgets.QInputDialog.getInt(self, "Input K", "Masukkan jumlah kluster (k):", 2, 1, 10, 1)
            
            if ok and isinstance(k, int):
                # Ubah gambar ke format yang dapat digunakan oleh K-Means (reshape jadi array 2D)
                pixel_values = self.image.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)

                # Definisikan kriteria K-Means
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

                # Terapkan K-Means clustering
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # Rekonstruksi gambar dari hasil clustering
                centers = np.uint8(centers)
                labels = labels.flatten()
                segmented_image = centers[labels]
                segmented_image = segmented_image.reshape(self.image.shape)

                self.processed_image = segmented_image
                self.display_image_in_view(self.processed_image)
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Input K tidak valid.")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or crop an image first.")

    def watershed_segmentation(self):
        if self.image is not None:
            # Konversi gambar ke grayscale dan terapkan threshold
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Gunakan distance transform untuk mendapatkan gambar yang lebih baik untuk marker
            dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(binary_image, sure_fg)

            # Label connected components
            _, markers = cv2.connectedComponents(sure_fg)

            # Tambahkan 1 ke semua marker agar background bukan 0
            markers = markers + 1

            # Tandai area yang tidak diketahui
            markers[unknown == 255] = 0

            # Terapkan watershed
            markers = cv2.watershed(self.image, markers)
            self.image[markers == -1] = [0, 0, 255]

            self.processed_image = self.image
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or crop an image first.")

    def global_thresholding(self):
        if self.image is not None:
            # Konversi gambar ke grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Terapkan threshold global
            _, thresholded_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            self.processed_image = thresholded_image
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or crop an image first.")

    def adaptive_thresholding(self):
        if self.image is not None:
            # Konversi gambar ke grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Terapkan adaptive thresholding
            adaptive_thresh_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY, 11, 2)

            self.processed_image = adaptive_thresh_image
            self.display_image_in_view(self.processed_image)
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or crop an image first.")
            
    def erosion(self):
        if self.processed_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.processed_image = cv2.erode(self.processed_image, kernel, iterations=1)
            self.display_image_in_view(self.processed_image)

    def dilation(self):
        if self.processed_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=1)
            self.display_image_in_view(self.processed_image)

    def opening(self):
        if self.processed_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel)
            self.display_image_in_view(self.processed_image)

    def closing(self):
        if self.processed_image is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
            self.display_image_in_view(self.processed_image)

    def hitormiss(self):
        if self.processed_image is not None:
            try:
                # Pastikan gambar dalam format grayscale atau biner
                if len(self.processed_image.shape) == 3:
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image

                # Konversi ke gambar biner
                _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                # Kernel untuk operasi Hit or Miss (harus berukuran sesuai)
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

                # Lakukan operasi Hit or Miss
                hitmiss_result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)

                # Pastikan operasi berhasil
                if hitmiss_result is None:
                    raise ValueError("Hit or Miss operation failed.")

                # Simpan hasilnya dan tampilkan
                self.processed_image = hitmiss_result
                self.display_image_in_view(self.processed_image)
            
            except Exception as e:
                print(f"Error in hitormiss: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Hit or Miss operation failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")

    def thinning(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah dalam format grayscale atau BGR
                if len(self.processed_image.shape) == 3:  # Jika gambar memiliki 3 channel (BGR)
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image  # Jika sudah grayscale

                # Konversi ke gambar biner dengan threshold invers
                _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

                # Operasi thinning menggunakan cv2.ximgproc.thinning
                self.processed_image = cv2.ximgproc.thinning(binary_image)

                # Tampilkan hasil thinning
                self.display_image_in_view(self.processed_image)

            except cv2.error as e:
                print(f"OpenCV Error in thinning: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Thinning operation failed: {e}")

            except Exception as e:
                print(f"Error in thinning: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Thinning operation failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")

    def skeletonization(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah dalam format grayscale atau BGR
                if len(self.processed_image.shape) == 3:  # Jika gambar memiliki 3 channel (BGR)
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image  # Jika sudah grayscale

                # Konversi ke gambar biner dengan threshold invers
                _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

                # Operasi skeletonization menggunakan cv2.ximgproc.thinning
                self.processed_image = cv2.ximgproc.thinning(binary_image)

                # Tampilkan hasil skeletonization
                self.display_image_in_view(self.processed_image)

            except cv2.error as e:
                print(f"OpenCV Error in skeletonization: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Skeletonization operation failed: {e}")

            except Exception as e:
                print(f"Error in skeletonization: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Skeletonization operation failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")

    def pruning(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah dalam format grayscale atau BGR
                if len(self.processed_image.shape) == 3:  # Jika gambar memiliki 3 channel (BGR)
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image  # Jika sudah grayscale

                # Konversi ke gambar biner dengan threshold invers
                _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

                # Operasi pruning (morfologi)
                kernel = np.ones((5, 5), np.uint8)  # Kernel untuk morfologi
                self.processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

                # Tampilkan hasil pruning
                self.display_image_in_view(self.processed_image)

            except cv2.error as e:
                print(f"OpenCV Error in pruning: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Pruning operation failed: {e}")

            except Exception as e:
                print(f"Error in pruning: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Pruning operation failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")
            
    def manual_convolution(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah dalam format grayscale atau BGR
                if len(self.processed_image.shape) == 3:  # Jika gambar memiliki 3 channel (BGR)
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image  # Jika sudah grayscale

                # Kernel rata-rata (Average filter) 3x3
                kernel = np.array([
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9]
                ])

                # Dimensi citra
                image_height, image_width = gray.shape
                kernel_height, kernel_width = kernel.shape

                # Padding untuk mengatasi piksel di tepi
                pad_h = kernel_height // 2
                pad_w = kernel_width // 2
                padded_image = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

                # Inisialisasi output array
                output = np.zeros_like(gray)

                # Konvolusi manual
                for i in range(image_height):
                    for j in range(image_width):
                        window = padded_image[i:i + kernel_height, j:j + kernel_width]
                        output[i, j] = np.sum(window * kernel)

                # Tampilkan citra asli di graphicsView
                if self.graphicsView is not None:
                    self.display_image_in_view(gray, self.graphicsView)  # Menampilkan citra asli

                # Tampilkan hasil konvolusi di graphicsView_2 jika ada, atau di graphicsView jika tidak ada graphicsView_2
                if self.graphicsView_2 is not None:
                    self.display_image_in_view(output, self.graphicsView_2)  # Menampilkan hasil konvolusi manual
                else:
                    self.display_image_in_view(output, self.graphicsView)  # Menampilkan hasil konvolusi manual jika hanya ada satu view

            except Exception as e:
                print(f"Error in manual_convolution: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"Manual convolution failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")

    def opencv_convolution(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah dalam format grayscale atau BGR
                if len(self.processed_image.shape) == 3:  # Jika gambar memiliki 3 channel (BGR)
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image  # Jika sudah grayscale

                # Kernel rata-rata (Average filter) 3x3
                kernel = np.ones((3, 3), np.float32) / 9

                # Lakukan konvolusi menggunakan OpenCV
                output = cv2.filter2D(gray, -1, kernel)

                # Tampilkan citra asli di graphicsView
                if self.graphicsView is not None:
                    self.display_image_in_view(gray, self.graphicsView)  # Menampilkan citra asli

                # Tampilkan hasil konvolusi di graphicsView_2 jika ada, atau di graphicsView jika tidak ada graphicsView_2
                if self.graphicsView_2 is not None:
                    self.display_image_in_view(output, self.graphicsView_2)  # Menampilkan hasil konvolusi dengan OpenCV
                else:
                    self.display_image_in_view(output, self.graphicsView)  # Menampilkan hasil konvolusi di graphicsView jika hanya satu view

            except cv2.error as e:
                print(f"OpenCV Error in opencv_convolution: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"OpenCV convolution failed: {e}")

            except Exception as e:
                print(f"Error in opencv_convolution: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"OpenCV convolution failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")
            
    def citra_warna_rgb(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar memiliki 3 channel (RGB/BGR)
                if len(self.processed_image.shape) == 3:
                    # Pisahkan channel R, G, B
                    blue, green, red = cv2.split(self.processed_image)

                    # Tampilkan citra inputan asli di graphicsView
                    self.display_image_in_view(self.processed_image, self.graphicsView)

                    # Hitung intensitas rata-rata masing-masing channel
                    red_mean = np.mean(red)
                    green_mean = np.mean(green)
                    blue_mean = np.mean(blue)

                    # Buat pop-up untuk menampilkan informasi intensitas rata-rata masing-masing channel
                    rgb_info = f"Red Channel Mean: {red_mean:.2f}\nGreen Channel Mean: {green_mean:.2f}\nBlue Channel Mean: {blue_mean:.2f}"
                    QtWidgets.QMessageBox.information(self, "RGB Channel Information", rgb_info)

                    # Simpan hasil ke Excel
                    data_rgb = {'Channel': ['Red', 'Green', 'Blue'],
                                'Mean Intensity': [red_mean, green_mean, blue_mean]}
                    df_rgb = pd.DataFrame(data_rgb)

                    # Tentukan folder penyimpanan
                    folder = "hasil_rgb_glcm"
                    os.makedirs(folder, exist_ok=True)  # Buat folder jika belum ada
                    file_path = os.path.join(folder, "rgb_analysis.xlsx")

                    # Simpan DataFrame ke file Excel
                    df_rgb.to_excel(file_path, index=False)
                    QtWidgets.QMessageBox.information(self, "Save Successful", f"RGB analysis saved to {file_path}")

                else:
                    QtWidgets.QMessageBox.warning(self, "Warning", "Input image is not in RGB format.")

            except Exception as e:
                print(f"Error in extract_rgb_channels: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"RGB channel extraction failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")

    def tekstur_dengan_glcm(self):
        if self.processed_image is not None:
            try:
                # Cek apakah gambar sudah grayscale
                if len(self.processed_image.shape) == 3:
                    gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.processed_image

                # Tampilkan citra grayscale di graphicsView
                self.display_image_in_view(gray, self.graphicsView)

                # Menghitung GLCM (Gray Level Co-occurrence Matrix)
                glcm = skft.greycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

                # Ekstraksi fitur GLCM (contrast, dissimilarity, homogeneity, energy, correlation)
                contrast = skft.greycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = skft.greycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = skft.greycoprops(glcm, 'homogeneity')[0, 0]
                energy = skft.greycoprops(glcm, 'energy')[0, 0]
                correlation = skft.greycoprops(glcm, 'correlation')[0, 0]

                # Menampilkan hasil GLCM di pop-up
                glcm_info = (f"GLCM Features:\n\n"
                            f"Contrast: {contrast:.2f}\n"
                            f"Dissimilarity: {dissimilarity:.2f}\n"
                            f"Homogeneity: {homogeneity:.2f}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Correlation: {correlation:.2f}")
                QtWidgets.QMessageBox.information(self, "GLCM Texture Analysis", glcm_info)

                # Simpan hasil ke Excel
                data_glcm = {'Feature': ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
                            'Value': [contrast, dissimilarity, homogeneity, energy, correlation]}
                df_glcm = pd.DataFrame(data_glcm)

                # Tentukan folder penyimpanan
                folder = "hasil_rgb_glcm"
                os.makedirs(folder, exist_ok=True)  # Buat folder jika belum ada
                file_path = os.path.join(folder, "glcm_analysis.xlsx")

                # Simpan DataFrame ke file Excel
                df_glcm.to_excel(file_path, index=False)
                QtWidgets.QMessageBox.information(self, "Save Successful", f"GLCM analysis saved to {file_path}")

            except Exception as e:
                print(f"Error in extract_texture_glcm: {e}")
                QtWidgets.QMessageBox.warning(self, "Error", f"GLCM extraction failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please open or process an image first.")
            
    def clear_output(self):
        """Hanya menghapus konten di graphicsView_2."""
        scene = QtWidgets.QGraphicsScene()  # Membuat scene baru yang kosong
        self.graphicsView_2.setScene(scene)  # Mengatur scene baru yang kosong ke graphicsView_2
        self.graphicsView_2.update()  # Memperbarui tampilan graphicsView_2
        print("graphicsView_2 has been cleared.")

    def clear_all(self):
        """Menghapus konten di graphicsView dan graphicsView_2."""
        scene_empty = QtWidgets.QGraphicsScene()  # Membuat scene baru yang kosong
        
        # Menghapus konten di graphicsView
        self.graphicsView.setScene(scene_empty)
        self.graphicsView.update()  # Memperbarui tampilan graphicsView
        
        # Menghapus konten di graphicsView_2
        self.graphicsView_2.setScene(scene_empty)
        self.graphicsView_2.update()  # Memperbarui tampilan graphicsView_2
        
        print("Both graphicsView and graphicsView_2 have been cleared.")

    def plot_histogram(self, image, title="Histogram", subplot=1):
        plt.subplot(1, 2, subplot)
        if len(image.shape) == 2:  # Grayscale image
            plt.hist(image.ravel(), 256, [0, 256])
        else:  # RGB image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.hist(image[..., i].ravel(), 256, [0, 256], color=color)
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim([0, 256])

    def show_tentang_dialog(self):
        # Create a new dialog and load the 'Tentang.ui' file
        tentang_dialog = QtWidgets.QDialog()
        uic.loadUi('Tentang.ui', tentang_dialog)
        tentang_dialog.exec_()  # Display the 'Tentang' dialog
        
    def show_arithmetical_dialog(self):
        self.arithmetical_window = ArithmeticalWindow()
        self.arithmetical_window.show()
        
    def set_bit_depth(self, bit_depth):
        self.bit_depth = bit_depth

    def display_image_in_view(self, image):
        # Convert numpy image to QImage
        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            q_image = QtGui.QImage(image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        else:  # Color image
            height, width, channel = image.shape
            q_image = QtGui.QImage(image.data, width, height, width * channel, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(q_image)

        # Create a QGraphicsScene for the image
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_2.setScene(scene)
        self.graphicsView_2.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
