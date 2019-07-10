import os
from PyQt5 import QtCore, QtGui, QtWidgets
from main import execute, training, loadModel

class basicWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        #Define grid layout
        grid_layout = QtWidgets.QGridLayout()
        self.setLayout(grid_layout)

        #Row 0 widgets
        self.modelLabel = QtWidgets.QLabel("Select your model:")

        self.modelCombo = QtWidgets.QComboBox()
        self.modelCombo.addItems(["SqueezeNet", "VGG16"])
        self.modelCombo.activated.connect(self.changeModel)  

        self.pretrainedCheckBox = QtWidgets.QCheckBox("Pretrained")
        self.dataAugmentationCheckBox = QtWidgets.QCheckBox("Data Augmentation")

        self.epochsLabel = QtWidgets.QLabel("Epochs:")

        self.epochsInput = QtWidgets.QLineEdit(self)
        self.epochsInput.setFixedWidth(120)
        # Only allow 2 digits not starting with a zero
        regex = QtCore.QRegExp('^[1-9]\d{1}$')
        validator = QtGui.QRegExpValidator(regex)
        self.epochsInput.setValidator(validator)
        self.epochsInput.textChanged.connect(self.enableTraining)

        self.trainingButton = QtWidgets.QPushButton('Train', self)
        self.trainingButton.clicked.connect(self.trainModel)
        self.trainingButton.setEnabled(False)

        self.accuracyLabel = QtWidgets.QLabel("Accuracy:")

        self.accuracyValue = QtWidgets.QLabel()

        #Row 1 widgets
        self.selectImageButton = QtWidgets.QPushButton("Select Image")
        self.selectImageButton.clicked.connect(self.setImage)
        
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setFrameShape(QtWidgets.QFrame.Box)

        #Row 3 widgets
        self.predictedLabel = QtWidgets.QLabel("Predicted class")

        self.predictedValue = QtWidgets.QLabel()

        self.developerCheckBox = QtWidgets.QCheckBox("Developer Mode")
        self.developerCheckBox.stateChanged.connect(self.developerMode)

        #Row 4 widgets
        self.submitButton = QtWidgets.QPushButton('Submit', self)
        self.submitButton.clicked.connect(self.submitData)
        self.submitButton.setEnabled(False)

        #Add widgets on grid layout
        grid_layout.addWidget(self.modelLabel, 0, 0)
        grid_layout.addWidget(self.modelCombo, 0, 1)
        grid_layout.addWidget(self.pretrainedCheckBox, 0, 2)
        grid_layout.addWidget(self.dataAugmentationCheckBox, 0, 3)
        grid_layout.addWidget(self.epochsLabel, 0, 4, QtCore.Qt.AlignRight)
        grid_layout.addWidget(self.epochsInput, 0, 5)
        grid_layout.addWidget(self.trainingButton, 0, 6, QtCore.Qt.AlignRight)
        grid_layout.addWidget(self.accuracyLabel, 0, 7, QtCore.Qt.AlignRight)
        grid_layout.addWidget(self.accuracyValue, 0, 8)
        grid_layout.addWidget(self.selectImageButton, 1, 0, 2, 3)
        grid_layout.addWidget(self.imageLabel, 1, 3, 2, 6)
        grid_layout.addWidget(self.predictedLabel, 3, 0)
        grid_layout.addWidget(self.predictedValue, 3, 3)
        grid_layout.addWidget(self.developerCheckBox, 3, 8, QtCore.Qt.AlignRight)
        grid_layout.addWidget(self.submitButton, 4, 0, 1, 9)

        #Set Window config
        self.setFixedSize(1300, 800)
        self.setWindowTitle('ML - Button Demo')
        self.setWindowIcon(QtGui.QIcon('../img/logo.png'))

        #Uncheck developer mode
        self.developerMode(False)
        
    
    def changeModel(self):
        self.enableTraining()
        self.enableSubmit()

    def enableTraining(self):
        if self.epochsInput.text():
            self.trainingButton.setText("Train")
            self.trainingButton.setEnabled(True)
        else:
            self.trainingButton.setEnabled(False)

    def trainModel(self):
        self.trainingButton.setText("Trained")
        self.trainingButton.setEnabled(False)
        accuracy,_ = training(self.modelCombo.currentText(), self.dataAugmentationCheckBox.isChecked(), self.pretrainedCheckBox.isChecked(), int(self.epochsInput.text()))
        self.accuracyValue.setText(str(accuracy) + '%')
    
    def setImage(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if self.fileName: # If the user gives a file
            pixmap = QtGui.QPixmap(self.fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLabel.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.predictedValue.setText('')
            self.enableSubmit()
        else:
            self.imageLabel.setPixmap(QtGui.QPixmap())
            self.submitButton.setEnabled(False)
    
    def submitData(self):
        predictedClass = execute(self.modelCombo.currentText(),self.fileName)
        self.predictedValue.setText(str(predictedClass))
        self.submitButton.setEnabled(False)
        
    def developerMode(self, state):
        if (QtCore.Qt.Checked == state):
            self.pretrainedCheckBox.show()
            self.dataAugmentationCheckBox.show()
            self.epochsLabel.show()
            self.epochsInput.show()
            self.trainingButton.show()
            self.accuracyLabel.show()
            self.accuracyValue.show()
        else:
            self.pretrainedCheckBox.hide()
            self.dataAugmentationCheckBox.hide()
            self.epochsLabel.hide()
            self.epochsInput.hide()
            self.trainingButton.hide()
            self.accuracyLabel.hide()
            self.accuracyValue.hide()
    
    def enableSubmit(self):
        if (os.path.isfile('checkpoint\\' + self.modelCombo.currentText() + '_checkpoint.pth')):
            self.submitButton.setText("Submit")
            if self.imageLabel.pixmap():
                self.submitButton.setEnabled(True)
        else:
            self.submitButton.setText("Train your model")
            self.submitButton.setEnabled(False)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    windowExample = basicWindow()
    windowExample.show()
    sys.exit(app.exec_())