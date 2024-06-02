# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:56:40 2024

@author: Administrator
"""

import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QPlainTextEdit, QApplication, QMainWindow, QLabel, QFileDialog, QTabWidget, QWidget, QVBoxLayout, QStatusBar, QScrollArea, QGridLayout, QPushButton
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QImage

from timm import create_model
import torch
import torchvision.transforms as T

from wildlife_tools.data import FeatureDatabase
from wildlife_tools.inference import KnnMatcher

from PIL import Image
import os
import numpy as np
 
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.intiUI()
    
    def intiUI(self):
        self.setWindowTitle("Giraffe Re-ID Tool")
        self.setGeometry(100, 100, 800, 600)
        self.message = []
 
        # 创建选项卡窗口
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)
 
        # 创建图像选项卡
        # self.image_tab = QWidget()
        # self.image_layout = QVBoxLayout(self.image_tab)
        # self.image_label = QLabel(self.image_tab)
        # self.image_layout.addWidget(self.image_label)
        # self.tabs.addTab(self.image_tab, "Show Image")
        
        # 创建滚动区域
        self.scroll_area_images = QScrollArea(self)
        self.scroll_area_images.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget(self)
        self.scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContents')
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scroll_area_images.setWidget(self.scrollAreaWidgetContents)
        self.scroll_area_images.setGeometry(20, 70, 600, 350)
        self.vertocall = QVBoxLayout()
        self.vertocall.addWidget(self.scroll_area_images)
        self.image_label = QLabel(self.scroll_area_images)
        self.gridLayout.addWidget(self.image_label)
        
        # 创建文本显示区域
        self.view_area = QPlainTextEdit(self)
        self.view_area.setPlaceholderText('Show Prediction Results Here')
        self.view_area.setGeometry(20, 430, 600, 130)
        
        
        # 添加按键
        # Crop Image
        self.crop_pushbutton = QPushButton(self)
        self.crop_pushbutton.setGeometry(650, 250, 100, 30)
        self.crop_pushbutton.setObjectName('crop_pushbutton')
        self.crop_pushbutton.setText('Preprocess')
#        self.open_file_pushbutton.clicked.connect(self.open)##关联函数

        # 添加按键
        # Prediction
        self.prediction_pushbutton = QPushButton(self)
        self.prediction_pushbutton.setGeometry(650, 470, 100, 30)
        self.prediction_pushbutton.setObjectName('prediction_pushbutton')
        self.prediction_pushbutton.setText('Predict ID')
        self.prediction_pushbutton.clicked.connect(self.algorithm)

        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
    def algorithm(self):
        print(self.message)
        if os.path.isfile(self.message):
            img = Image.open(self.message)
            output = model(train_transforms(img).unsqueeze(0))
            pred_item, score = matcher([output])
            print(pred_item, score)
            self.view_area.setPlainText(f"ID: {pred_item}, Accuracy: {score}") 
        elif os.path.isdir(self.message):
            folder_path = self.message
            self.view_area.setPlainText("")
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path).convert('RGB')
                    # print(np.array(img).shape)
                    output = model(train_transforms(img).unsqueeze(0))
                    pred_item, score = matcher([output])
                    print(pred_item, score)
                    self.view_area.appendPlainText(f"ID: {pred_item}, Accuracy: {score}")
        else:
            return 'none'
        

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", ".", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)
            scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.message = filename
            self.statusBar.showMessage(filename)
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.display_image_size = 80
        self.col=0
        self.row=0
        # self.start_img_viewer(folder_path)
        self.display_images(folder_path)
        self.message = folder_path
        self.statusBar.showMessage(folder_path)
    
    def clear_layout(self):
        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deletelater()
            
    def start_img_viewer(self, folder_path):
        file_count = 0
        print(folder_path)
        if folder_path:
            for filename in os.listdir(folder_path):
                ext = os.path.splitext(filename)[1]
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    file_count = file_count + 1
            if file_count != 0:
                print('file numbers ', file_count)
                for filename in os.listdir(folder_path):
                    ext = os.path.splitext(filename)[1]
                    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_path = os.path.join(folder_path, filename)
                        image = QImage(image_path)
                        pixmap = QPixmap.fromImage(image)
                        self.addImage(pixmap, filename)
                        QApplication.processEvents()##实时加载，可能图片加载数量比较多
            else:
                QMessageBox.information(self, 'Message', 'The folder is empty')
        else:
            QMessageBox.information(self, 'Message', 'Please select image folder')
                
    def get_nr_of_image_columns(self):
        #展示图片的区域，计算每排显示图片数。返回的列数-1是因为我不想频率拖动左右滚动条，影响数据筛选效率
        scroll_area_images_width = int(0.68*self.width())
        if scroll_area_images_width > self.display_image_size:
            pic_of_columns = scroll_area_images_width // self.display_image_size  #计算出一行几列；
        else:
            pic_of_columns = 1
 
        return pic_of_columns-1
 
    def addImage(self, pixmap, image_id):
        ##获取图片列数
        nr_of_columns = self.get_nr_of_image_columns()
        nr_of_widgets = self.gridLayout.count()
        # print('列数', nr_of_columns)
        self.max_columns = nr_of_columns
        if self.col < self.max_columns:
            self.col += 1
        else:
            self.col = 0
            self.row += 1
        # clickable_image = QClickableImage(self.display_image_size, self.display_image_size, pixmap, image_id)
        scaled_pixmap = pixmap.scaled(self.display_image_size, self.display_image_size, aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        # self.gridLayout.addWidget(scaled_pixmap, self.row, self.col)
        
           
    def display_images(self, folder_path):
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1]
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = os.path.join(folder_path, filename)
                image = QImage(image_path)
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
                self.image_label.setPixmap(scaled_pixmap)
                self.statusBar.showMessage(folder_path)
                break  # 只显示一张图片，如需显示所有图片，移除此行并在display_images中使用循环

    def imageId(self):
        return self.image_id


 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
 
    # 添加菜单栏和工具栏
    menubar = window.menuBar()
    file_menu = menubar.addMenu("File")
    image_action = file_menu.addAction("Open Image")
    folder_action = file_menu.addAction("Select Folder")
 
    toolbar = window.addToolBar("File")
    toolbar.addAction(image_action)
    toolbar.addAction(folder_action)
 
    # 连接信号和槽
    image_action.triggered.connect(window.open_image)
    folder_action.triggered.connect(window.load_folder)
    
    # 加载算法
    local_weight_path = 'checkpoint/checkpoint_SL_Patch4_7_224_R_Giraffe_Chest_epoch100.pth'
    model = create_model('swin_large_patch4_window7_224', checkpoint_path = local_weight_path, num_classes=0, pretrained=True, pretrained_cfg_overlay=dict(file="MegaDescriptor-L-224/pytorch_model.bin"))
    model = model.eval()

    train_transforms = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    database = FeatureDatabase.from_file('DeepFeature/DeepFeatures_R_Giraffe_Chest.pkl')
    matcher = KnnMatcher(database)
 
    sys.exit(app.exec_())