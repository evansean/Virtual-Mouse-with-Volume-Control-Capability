import cv2
import time
import os
import HandTrackingModule as htm

widthCam,heightCam = 640,480
prevTime = 0
cap = cv2.VideoCapture(0)
cap.set(3,widthCam)
cap.set(4,heightCam)

detector = htm.handDetection(detectionConfidence=0.75)

tipIds = [4,8,12,16,20] #tip id of thumb, index, middle, ring, pinky finger respectively

while True:
    sucess,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPos(img,draw=False)
    handPres = detector.handPresence(img)
    # print(handPres)
    # print(lmList)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[1] and fingers[2]:
            print("Scroll Up")
        if fingers[1] and fingers[2]==False:
            print("Scroll Down")
    #Calculate FPS
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)