# -*- coding: utf-8 -*-
"""
Calculates the 3x3 matrix to transform the four source points to the four destination points

Comment copied from OpenCV:
/* Calculates coefficients of perspective transformation
* which maps soruce (xi,yi) to destination (ui,vi), (i=1,2,3,4):
*
*      c00*xi + c01*yi + c02
* ui = ---------------------
*      c20*xi + c21*yi + c22
*
*      c10*xi + c11*yi + c12
* vi = ---------------------
*      c20*xi + c21*yi + c22
*
* Coefficients are calculated by solving linear system:
*             a                         x    b
* / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
* | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
* | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
* | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
* |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
* |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
* |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
* \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
*
* where:
*   cij - matrix coefficients, c22 = 1
*/

"""

# 引入套件
import cv2
import numpy as np


path = 'img/view.jpg' # 圖片路徑


def show(image):
    # 顯示圖片
    cv2.imshow('image', image) 
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0) # 若設定為 0 就表示持續等待至使用者按下按鍵為止
    cv2.destroyAllWindows()


def pointInPolygon(x, y, point):
    j = len(point) - 1
    flag = False
    for i in range(len(point)):
        if (point[i][1] < y <= point[j][1] or point[j][1] < y <= point[i][1]) and (point[i][0] <= x or point[j][0] <= x):
            if point[i][0] + (y - point[i][1]) / (point[j][1] - point[i][1]) * (point[j][0] - point[i][0]) < x:
                flag = not flag
        j = i
    return flag

def draw_line(image, point, color=(0, 255, 0), w=2):
    image = cv2.line(image, (point[0][0], point[0][1]), (point[1][0], point[1][1]), color, w)
    image = cv2.line(image, (point[1][0], point[1][1]), (point[2][0], point[2][1]), color, w)
    image = cv2.line(image, (point[2][0], point[2][1]), (point[3][0], point[3][1]), color, w)
    image = cv2.line(image, (point[3][0], point[3][1]), (point[0][0], point[0][1]), color, w)
    return image


def perspectiveLogo(image, point1, point2):
    h, w = image.shape[:2] # 獲取圖像的大小，shape返回的是一個tuple元組，第一個元素表示圖像的高度，第二個表示圖像的寬度，第三個表示像素的通道數。
    print(h, w) # (1108,1477)取高度和寬度就好 
    #print(point2[2][0]) # 300.0
    #print(point2[2][1]) # 300.0
    img1 = np.zeros((int(point2[2][0]), int(point2[2][1]), 3), dtype=np.uint8) # 給定300*300 長度3 shape形狀>用0填充數組
    M = cv2.getPerspectiveTransform(point1, point2) # 計算得到轉換的矩陣
    '''
        M
    [[ 1.55112036e-01 -6.37446723e-03 -2.02176852e+01]
     [-4.56013848e-02  3.22669109e-01 -4.60250059e+01]
     [-4.08228948e-04  7.07633919e-06  1.00000000e+00]]
    '''
    # 自己做透視變換
    for i in range(h):
        for j in range(w):
                x = (M[0][0]*j + M[0][1]*i + M[0][2]) / (M[2][0]*j + M[2][1]*i + M[2][2]) + 0.5
                y = (M[1][0]*j + M[1][1]*i + M[1][2]) / (M[2][0]*j + M[2][1]*i + M[2][2]) + 0.5
                x, y = int(x), int(y)
                #print(x, y)
                if 1 <= x < point2[2][0]-1 and 1 <= y < point2[2][1]-1:
                    img1[y, x, :] = image[i, j, :]
                    img1[y, x-1, :] = image[i, j, :]
                    img1[y, x+1, :] = image[i, j, :]
                    img1[y-1, x, :] = image[i, j, :]
                    img1[y+1, x, :] = image[i, j, :]
    img2 = cv2.warpPerspective(image, M, (300, 300)) # 利用CV2透視變換成是相對於原圖image是以M變換後的圖像
    
    img = np.hstack((img1, img2)) # 水平(按列順序)把陣列給堆疊起來(兩者可以做比較)
    show(img)
    cv2.imwrite('output.jpg', img) # 寫入圖檔

def main():
    image = cv2.imread(path) # 加載圖像
    img = image.copy() # 複製圖片

    point1 = np.float32([[137, 162], [1163, 307], [1178, 797], [173, 1038]]) # 標出四個點
    #print(point1)
    point2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]]) # 輸出的四個點
    #print(point2)
    perspectiveLogo(image, point1, point2)

    img = draw_line(img, point1)
    show(img) # show 出我標的點的輪廓圖

if __name__ == '__main__':
    main()
