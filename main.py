#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints

# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））
def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders

# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(path, maxArea=50):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w*h > maxArea:
            border = [(x, y), (x+w, y+h)]
            borders.append(border)
    return borders


# 显示结果及边框
def showResults(path, borders, results=None):
    img = cv2.imread(path)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        #cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('test', img)
    cv2.waitKey(0)

# 根据边框转换为MNIST格式
def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

# 预测手写数字
def predict(modelpath, imgData):
    from keras import models
    my_mnist_model = models.load_model(modelpath)
    print(my_mnist_model.summary())
    img = imgData.astype('float32') / 255
    results = my_mnist_model.predict(img)
    result_number = []
    for result in results:
        result_number.append(np.argmax(result))
    return result_number

if __name__ == '__main__':
    # 综合测试
    # path = 'test1.png'
    path = 'test2.jpg'
    model = 'my_mnist_model_old.h5'
    borders = findBorderContours(path)
    imgData = transMNIST(path, borders)
    results = predict(model, imgData)
    showResults(path, borders, results)


    # # 测试代码 accesspiexl
    # path = 'test2.jpg'
    # img = cv2.imread(path, 0)
    # img = accessPiexl(img)
    # cv2.imshow('accesspiexl', img)
    # cv2.waitKey(0)

    # # 测试代码 accessBinary
    # path = 'test1.png'
    # img = cv2.imread(path, 0)
    # img = accessBinary(img)
    # cv2.imshow('accessBinary', img)
    # cv2.waitKey(0)

    # 测试代码 findBorderHistogram
    # path = 'test2.jpg'
    # borders = findBorderHistogram(path)
    # showResults(path, borders)

    # 测试代码 findBorderContours
    # path = 'wenzi.jpg'
    # borders = findBorderContours(path)
    # print(borders)
    # showResults(path, borders)

    # # 测试代码 transMNIST
    # path = 'test2.jpg'
    # borders = findBorderContours(path)
    # imgData = transMNIST(path, borders)
    # for i, img in enumerate(imgData):
    #     # cv2.imshow('test', img)
    #     # cv2.waitKey(0)
    #     name = 'extract/test_' + str(i) + '.jpg'
    #     cv2.imwrite(name, img)
