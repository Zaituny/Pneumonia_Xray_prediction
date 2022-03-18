import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui  import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from PyQt5.uic import loadUi
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		loadUi('pneumo_predict.ui', self)
		self.load_image_button.clicked.connect(self.load_func)
		self.predict_button.clicked.connect(self.predict_func)
	def load_func(self):
		fname = QFileDialog.getOpenFileName(self, 'File Explorer')
		pxm = QPixmap(fname[0])
		self.image_holder.resize(pxm.width(), pxm.height())
		self.image_holder.setPixmap(pxm)
		self.image_holder.repaint()
		self.image_path.setText(fname[0])
		self.img = image.load_img(fname[0], target_size = (256, 256, 3))
		self.img = image.img_to_array(self.img)
		self.img = np.expand_dims(self.img, axis=0)
                

	def predict_func(self):
		self.model = load_model('model3.h5')
		self.prediction = np.argmax(self.model.predict(self.img), axis = 1) 
		if self.prediction == 0:
			self.predicted_value.setText('Bacterial_Pneumonia')
		if self.prediction == 1:
			self.predicted_value.setText('Normal')
		if self.prediction == 2:
			self.predicted_value.setText('Viral_Pneumonia')
		self.confidence = round(100 * (np.max(self.model.predict(self.img))), 2)
		self.confidence_percentage.setText(str(self.confidence))

app = QApplication(sys.argv)
mainwindow = MainWindow()

widget = QtWidgets.QStackedWidget()

widget.addWidget(mainwindow)
widget.show()
sys.exit(app.exec_())

