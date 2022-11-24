import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


#-----VARIABLES-------
widthCam, heightCam = 640,480
widthScreen,heightScreen = autopy.screen.size() #1280,720
frameR = 100 #frame reduction
smooth = 5
pLocX,pLocY = 0,0
curLocx,curLocY =0,0
#-----VARIABLES-------
# print(widthScreen,heightScreen)
cap = cv2.VideoCapture(0)
#3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
cap.set(3,widthCam)
cap.set(4,heightCam)
prevTime=0
detector = htm.handDetection(detectionConfidence=0.7)
while True:
    # Find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPos(img)
    #Left click
    #Scroll up
    #Scroll down
    #Volume control


    # Get the index and middle finger tips
    if len(lmList) != 0:
        # coordinates of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
    # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR), (widthCam-frameR,heightCam-frameR),(255,0,255),2)

    # Moving mode: Only index finger up
        if fingers[1] == 1 and fingers[2] ==0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR,widthCam-frameR), (0,widthScreen)) 
            y3 = np.interp(y1,(frameR,heightCam-frameR), (0,heightScreen))
            # Smooth values
            curLocx = pLocX + (x3 -pLocX)/smooth
            curLocY = pLocY + (y3-pLocY)/smooth
            # Move mouse
            autopy.mouse.move(widthScreen- curLocx, curLocY)
            cv2.circle(img, (x1,y1),15,(255,0,255),cv2.FILLED)
            pLocX,pLocY = curLocx,curLocY
    # Clicking mode: both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] ==1:
                # Find distance between fingers
                length,img, lineInfo = detector.findDistance(8,12,img)
                if length < 50:
                    cv2.circle(img, (lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                    # Click mouse if distance is short
                    autopy.mouse.click()
    #calculate fps
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime
    
    # Display
    cv2.putText(img, f'FPS: {int(fps)} ',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)