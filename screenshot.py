import cv2,os
def doing_screenshot():
    cam = cv2.VideoCapture(0)

    for i in range(35):
        cam.read()

    ret,frame = cam.read()

    cv2.imwrite('camer.png',frame)

    cam.release()