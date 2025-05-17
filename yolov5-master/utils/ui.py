import os
import sys
from datetime import time

import cv2
import random

import qdarkstyle
import torch
import numpy as np
import torch.backends.cudnn as cudnn

import pymysql
from dbutils.pooled_db import PooledDB
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.dataloaders import letterbox
from utils.plots import plot_one_box


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True
        weights = 'D:\\Code\\yolov5\\yolov5-master\\runs\\train\\exp11\\weights\\last.pt'  # 模型加载路径
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.25  # NMS置信度
        self.iou_thres = 0.45  # IOU阈值

        # 载入模型
        self.model = attempt_load(weights, device=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()  # to FP16

        # 从模型中获取各类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # 给每一个类别初始化颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        # MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.centralwidget.setStyleSheet("border: 1px solid white;")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        # self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        # 打开图片按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 40))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)  # 增加垂直盒子内部对象间距

        # 打开摄像头按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(5)

        # 打开视频按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 40))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 40))
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(50)

        # 显示导出文件夹按钮
        self.pushButton_showdir = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_showdir.sizePolicy().hasHeightForWidth())
        self.pushButton_showdir.setSizePolicy(sizePolicy)
        self.pushButton_showdir.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_showdir.setMaximumSize(QtCore.QSize(150, 50))
        self.pushButton_showdir.setFont(font)
        self.pushButton_showdir.setObjectName("pushButton_showdir")
        self.verticalLayout.addWidget(self.pushButton_showdir, 0, QtCore.Qt.AlignHCenter)

        # 右侧图片/视频填充区域
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.label.setStyleSheet("border: 1px solid white;")  # 添加显示区域边框

        # 底部美化导航条
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv5目标检测平台"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_showdir.setText(_translate("MainWindow", "打开输出文件夹"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_showdir.clicked.connect(self.button_show_dir)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('')   # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        print('打开图片')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return
        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.imgsz)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        # print(label.split()[0])  # 打印各目标名称
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite('result/prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('result/vedio_prediction.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('result/camera_prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):
        name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            # print(label)  # 打印各目标+置信度
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()

    def button_show_dir(self):
        path = os.getcwd() + '\\' + 'result'
        os.system(f"start explorer {path}")

    def save_detection_to_db(self, detection_info):
        # detection_info 应该是一个包含检测信息的字典，例如：
        # detection_info = {
        #     'class_name': class_name,
        #     'confidence': confidence,
        #     'xyxy': xyxy
        # }
        x1, y1, x2, y2 = [int(item) for item in detection_info['xyxy']]
        SQLHelper.fetch_one(
            "INSERT INTO detections (class_name, confidence, x1, y1, x2, y2, detection_time) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (detection_info['class_name'], detection_info['confidence'], x1, y1, x2, y2, detection_info['time'])
        )


# 数据库配置模块
class Config(object):
    # DEBUG = True
    # SECRET_KEY = "umsuldfsdflskjdf"
    # PERMANENT_SESSION_LIFETIME = timedelta(minutes=20)
    # SESSION_REFRESH_EACH_REQUEST = True
    # SESSION_TYPE = "redis"
    PYMYSQL_POOL = PooledDB(
        creator=pymysql,  # 使用链接数据库的模块
        maxconnections=6,  # 连接池允许的最大连接数，0和None表示不限制连接数
        mincached=2,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
        maxcached=5,  # 链接池中最多闲置的链接，0和None不限制
        maxshared=3,
        # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
        blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
        maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
        setsession=[],  # 开始会话前执行的命令列表。
        ping=0,
        # ping MySQL服务端，检查是否服务可用。
        host='localhost',
        port=3306,
        user='root',
        password='060811',  # 你自己的密码
        database='demo',  # 提前建好库
        charset='utf8'
    )


# 数据库操作类
class SQLHelper(object):
    @staticmethod
    def open(cursor):
        POOL = Config.PYMYSQL_POOL
        conn = POOL.connection()
        cursor = conn.cursor(cursor=cursor)
        return conn, cursor

    @staticmethod
    def close(conn, cursor):
        conn.commit()
        cursor.close()
        conn.close()

    @classmethod
    def fetch_one(cls, sql, args, cursor=pymysql.cursors.DictCursor):
        conn, cursor = cls.open(cursor)
        cursor.execute(sql, args)
        obj = cursor.fetchone()
        cls.close(conn, cursor)
        return obj

    @classmethod
    def fetch_all(cls, sql, args, cursor=pymysql.cursors.DictCursor):
        conn, cursor = cls.open(cursor)
        cursor.execute(sql, args)
        obj = cursor.fetchall()
        cls.close(conn, cursor)
        return obj

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口透明度
    # ui.setWindowOpacity(0.93)
    # 去除顶部边框
    # ui.setWindowFlags(Qt.FramelessWindowHint)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    ui.setWindowIcon(icon)
    ui.show()
    sys.exit(app.exec_())