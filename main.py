import os
import cv2
import imutils
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/1/Desktop/COS_img/fotos/foto.jpg', 0)
image_to_find = cv2.imread('C:/Users/1/Desktop/COS_img/fotos/foto_to_find.jpg', 0)
rotated_image = cv2.imread('C:/Users/1/Desktop/COS_img/rotation/+/10.jpg', 0)
scaled_image = cv2.imread('C:/Users/1/Desktop/COS_img/scale/1+/1,1.jpg', 0)

def show_picture():
    pictures = os.listdir('C:/Users/1/Desktop/COS_img/fotos')
    pic_box = plt.figure(figsize=(10, 4))
    for i, picture in enumerate(pictures):
        picture = cv2.imread('C:/Users/1/Desktop/COS_img/fotos/' + picture)
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        pic_box.add_subplot(1, 2, i + 1)
        plt.imshow(picture)
        plt.axis('off')
    plt.show()

def show_results():
    img1 = image.copy()
    dimensions1 = image_to_find.shape
    res1 = cv2.matchTemplate(img1, image_to_find, cv2.TM_CCORR_NORMED)
    cv2.imshow('Korr', res1)
    (_, _, _, maxLoc1) = cv2.minMaxLoc(res1, None)
    cv2.rectangle(img1, maxLoc1, (maxLoc1[0] + dimensions1[1], maxLoc1[1] + dimensions1[0]), (255, 255, 255))
    cv2.imshow('Output', img1)
    cv2.imshow('To_find', image_to_find)

def rotation_correlation():
    img2 = image.copy()
    dimensions2 = rotated_image.shape
    res2 = cv2.matchTemplate(img2, rotated_image, cv2.TM_CCORR_NORMED)
    (_, _, _, maxLoc2) = cv2.minMaxLoc(res2, None)
    cv2.rectangle(img2, maxLoc2, (maxLoc2[0] + dimensions2[1], maxLoc2[1] + dimensions2[0]), (255, 255, 255))
    cv2.imshow('Output', img2)
    cv2.imshow('To_find', image_to_find)

def scaling_correlation():
    img3 = image.copy()
    dimensions3 = scaled_image.shape
    res3 = cv2.matchTemplate(img3, scaled_image, cv2.TM_CCORR_NORMED)
    (_, _, _, maxLoc3) = cv2.minMaxLoc(res3, None)
    cv2.rectangle(img3, maxLoc3, (maxLoc3[0] + dimensions3[1], maxLoc3[1] + dimensions3[0]), (255, 255, 255))
    cv2.imshow('Output', img3)
    cv2.imshow('To_find', image_to_find)

root = Tk()
root.title("lab1")
root.geometry("250x200")

t1=Button(text="Show picture", command=show_picture)
t1.place(width=175, relx=0.15, rely=0.15)
t2=Button(text="Find segment", command=show_results)
t2.place(width=175, relx=0.15, rely=0.3)
t3=Button(text="Rotation correlation", command=rotation_correlation)
t3.place(width=175, relx=0.15, rely=0.45)
t4=Button(text="Scaling correlation", command=scaling_correlation)
t4.place(width=175, relx=0.15, rely=0.6)

root.mainloop()
cv2.waitKey(0)