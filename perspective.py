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

import cv2
import numpy as np


path = 'img/view.jpg'


def show(image):
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('image', image)
    cv2.waitKey(0)
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


def warp(image, point1, point2):
    h, w = image.shape[:2]
    print(h, w)
    img1 = np.zeros((int(point2[2][0]), int(point2[2][1]), 3), dtype=np.uint8)
    M = cv2.getPerspectiveTransform(point1, point2)
    for i in range(h):
        for j in range(w):
            # if pointInPolygon(j, i, point1):
                x = (M[0][0]*j + M[0][1]*i + M[0][2]) / (M[2][0]*j + M[2][1]*i + M[2][2]) + 0.5
                y = (M[1][0]*j + M[1][1]*i + M[1][2]) / (M[2][0]*j + M[2][1]*i + M[2][2]) + 0.5
                x, y = int(x), int(y)
                # print(x, y)
                if 1 <= x < point2[2][0]-1 and 1 <= y < point2[2][1]-1:
                    img1[y, x, :] = image[i, j, :]
                    img1[y, x-1, :] = image[i, j, :]
                    img1[y, x+1, :] = image[i, j, :]
                    img1[y-1, x, :] = image[i, j, :]
                    img1[y+1, x, :] = image[i, j, :]
    img2 = cv2.warpPerspective(image, M, (300, 300))
    img = np.hstack((img1, img2))
    show(img)

def main():
    image = cv2.imread(path)
    img = image.copy()
    point1 = np.float32([[137, 162], [1163, 307], [1178, 797], [173, 1038]])
    point2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    warp(image, point1, point2)


    img = draw_line(img, point1)
    images = np.hstack((image, img))
    show(images)

if __name__ == '__main__':
    main()
