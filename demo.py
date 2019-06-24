from PyQt5 import QtCore, QtGui, QtWidgets
from net.main import execute, training

class basicWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        grid_layout = QtWidgets.QGridLayout()
        self.setLayout(grid_layout)

        #Row 0 widgets
        self.modelLabel = QtWidgets.QLabel("Select your model:")

        self.modelCombo = QtWidgets.QComboBox()
        self.modelCombo.addItems(["SqueezeNet", "VGG16"])

        self.pretrainedCheckBox = QtWidgets.QCheckBox("Pretrained")

        self.epochsLabel = QtWidgets.QLabel("Epochs:")

        self.epochsInput = QtWidgets.QLineEdit(self)
        self.epochsInput.setFixedWidth(240)
        self.onlyInt = QtGui.QIntValidator()
        self.epochsInput.setValidator(self.onlyInt)

        self.trainingButton = QtWidgets.QPushButton('Train', self)
        self.trainingButton.clicked.connect(self.trainModel)

        self.accuracyLabel = QtWidgets.QLabel("Accuracy:")

        self.accuracyValue = QtWidgets.QLabel()

        #Row 1 widgets
        self.selectImageBtn = QtWidgets.QPushButton("Select Image")
        self.selectImageBtn.clicked.connect(self.setImage)
        
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setFrameShape(QtWidgets.QFrame.Box)

        #Row 3 widgets
        self.predictedLabel = QtWidgets.QLabel("Predicted class")

        self.predictedValue = QtWidgets.QLabel()

        #Row 4 widgets
        self.submitButton = QtWidgets.QPushButton('Submit', self)
        self.submitButton.clicked.connect(self.submitData)
        self.submitButton.setEnabled(False)

        grid_layout.addWidget(self.modelLabel, 0, 0)
        grid_layout.addWidget(self.modelCombo, 0, 1)
        grid_layout.addWidget(self.pretrainedCheckBox, 0, 2)
        grid_layout.addWidget(self.epochsLabel, 0, 3)
        grid_layout.addWidget(self.epochsInput, 0, 4)
        grid_layout.addWidget(self.trainingButton, 0, 6)
        grid_layout.addWidget(self.accuracyLabel, 0, 7, QtCore.Qt.AlignRight)
        grid_layout.addWidget(self.accuracyValue, 0, 8)
        grid_layout.addWidget(self.selectImageBtn, 1, 0, 2, 3)
        grid_layout.addWidget(self.imageLabel, 1, 3, 2, 6)
        grid_layout.addWidget(self.predictedLabel, 3, 0)
        grid_layout.addWidget(self.predictedValue, 3, 3)
        grid_layout.addWidget(self.submitButton, 4, 0, 1, 9)

        grid_layout.setRowStretch(1, 2)

        self.setFixedSize(1300, 800)
        self.setWindowTitle('ML - Button Demo')
        self.setWindowIcon(QtGui.QIcon('./img/logo.png'))
        
    
    def trainModel(self):
        self.trainingButton.setText("Trained")
        self.trainingButton.setEnabled(False)
        accuracy = training(self.modelCombo.currentText(), self.pretrainedCheckBox.isChecked(), int(self.epochsInput.text()))
        self.accuracyValue.setText(str(accuracy) + '%')
    
    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLabel.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.fileName = fileName
            self.predictedValue.setText('')
            self.submitButton.setEnabled(True)
    
    def submitData(self):
        predictedClass = execute(self.fileName)
        self.predictedValue.setText(str(predictedClass))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    windowExample = basicWindow()
    windowExample.show()
    sys.exit(app.exec_())