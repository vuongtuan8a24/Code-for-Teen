import numpy as sp
import cv2

cap = cv2.VideoCapture
mask = cv2.imread("E:\\C4T\\Lesson6\\6.jpg")
cascade = cv2.CascadeClassifier("E:\\C4T\\Lesson6\\haarcascade_frontalface_alt2 (1).xml")
while True:
    ret, frame = cap.read()

    #gray img
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)


    # tao ham con
    def giu_mau(f, a):

        faces = cascade.detectMultiScale(a)

        for x, y, w, h in faces:
            # tao mot hinh chu nhat bao quanh mat
            cv2.rectangle(f, (x, y), (x + w, y + h), (0, 0, 255), 5)
            newmask = cv2.resize(mask, (w, h), cv2.INTER_CUBIC)
            a = []
            a = gray.shape

            for i in a[0]:
                for j in a[1]:
                    if a[i,j] != 0:
                        f[y:y + h, x:x + w, :] = f[y:y + h, x:x + w, :] - newmask

    giu_mau(frame,gray)
    cv2.imshow('anh',frame)

    b = cv2.waitKey(50)
    if b == ord('q'):
        break



