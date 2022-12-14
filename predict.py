# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predict.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1000, 600)
        self.listView = QtWidgets.QListView(Dialog)
        self.listView.setGeometry(QtCore.QRect(0, 0, 1000, 600))
        self.listView.setStyleSheet("background-color: rgb(238, 234, 217);")
        self.listView.setObjectName("listView")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(250, 90, 500, 50))
        self.label.setStyleSheet("font: 75 16pt \"Arial\";")
        self.label.setObjectName("label")
        self.radioButton = QtWidgets.QRadioButton(Dialog)
        self.radioButton.setGeometry(QtCore.QRect(350, 300, 115, 19))
        self.radioButton.setStyleSheet("QRadioButton {\n"
"    \n"
"    font: 75 11pt \"Consolas\";\n"
"}")
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Dialog)
        self.radioButton_2.setGeometry(QtCore.QRect(350, 340, 115, 19))
        self.radioButton_2.setStyleSheet("QRadioButton {\n"
"    \n"
"    font: 75 11pt \"Consolas\";\n"
"}")
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(60, 460, 72, 15))
        self.label_2.setStyleSheet("font: 11pt \"Arial\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(360, 460, 72, 15))
        self.label_3.setStyleSheet("font: 11pt \"Arial\";")
        self.label_3.setObjectName("label_3")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(150, 450, 171, 41))
        self.textBrowser.setStyleSheet("font: 11pt \"Consolas\"; padding-top:1px;")
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setGeometry(QtCore.QRect(450, 450, 181, 41))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_2.setStyleSheet("font: 11pt \"Consolas\"; padding-top:1px;")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(660, 460, 72, 15))
        self.label_4.setStyleSheet("font: 11pt \"Arial\";")
        self.label_4.setObjectName("label_4")
        self.textBrowser_3 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_3.setGeometry(QtCore.QRect(750, 450, 181, 41))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_3.setStyleSheet("font: 11pt \"Consolas\"; padding-top:1px;")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(500, 310, 130, 40))
        self.pushButton.setStyleSheet("font: 12pt \"Arial\";")
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(150, 180, 701, 61))
        self.lineEdit.setStyleSheet("border-radius: 10px;\n"
"font: 75 11pt \"Arial\";\n"
"padding-left: 10px;")
        self.lineEdit.setObjectName("lineEdit")
        self.commandLinkButton = QtWidgets.QCommandLinkButton(Dialog)
        self.commandLinkButton.setGeometry(QtCore.QRect(710, 240, 141, 31))
        self.commandLinkButton.setStyleSheet("padding-top: 0px;\n"
                                             "font: 11pt \"??????\";\n"
                                             "color: rgb(0, 0, 255);\n"
                                             "")
        self.commandLinkButton.setObjectName("commandLinkButton")

        self.retranslateUi(Dialog)
        self.lineEdit.textEdited['QString'].connect(Dialog.setInput)
        self.pushButton.clicked.connect(Dialog.predict)
        self.radioButton.toggled['bool'].connect(Dialog.setFastText)
        self.radioButton_2.toggled['bool'].connect(Dialog.setBert)
        self.commandLinkButton.clicked.connect(Dialog.randomSentence)
        self.textBrowser_4 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_4.setGeometry(QtCore.QRect(570, 550, 411, 41))
        self.textBrowser_4.setStyleSheet("background-color: rgb(238, 234, 217);\n"
                                         "border: 0;\n"
                                         "font: 87 11pt \"Consolas\";\n"
                                         "color: rgb(0, 0, 128);")
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.textBrowser_4.setAlignment(QtCore.Qt.AlignRight)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "??????FastText&Bert???????????????????????????"))
        self.radioButton.setText(_translate("Dialog", "FastText"))
        self.radioButton_2.setText(_translate("Dialog", "Bert"))
        self.label_2.setText(_translate("Dialog", "????????????"))
        self.label_3.setText(_translate("Dialog", "????????????"))
        self.label_4.setText(_translate("Dialog", "????????????"))
        self.pushButton.setText(_translate("Dialog", "????????????"))
        self.lineEdit.setPlaceholderText(_translate("Dialog", "????????????????????????????????????"))
        self.commandLinkButton.setText(_translate("Dialog", "???????????????"))
