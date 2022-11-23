import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################
widthCam, heightCam = 640,480

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange() #(-96.0, 0.0, 1.5)
minVol = volRange[0]
maxVol = volRange[1]
#####################

cap = cv2.VideoCapture(0)
#3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
cap.set(3,widthCam)
cap.set(4,heightCam)
prevTime=0
vol=0
volBar=400
volPer=0
detector = htm.handDetection(detectionConfidence=0.7)

while True:
    sucess,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPos(img,draw=False)
    if len(lmList)!= 0:
        # print(lmList[4],lmList[8])
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img,(x1,y1), 15,(255,0,255), cv2.FILLED)
        cv2.circle(img,(x2,y2), 15,(255,0,255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy), 15,(255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        # Hand range 15-200
        #Vol range -96.0 - 0.0

        #calibrate range in the future
        vol = np.interp(length, [0,260],[minVol,maxVol])
        volBar = np.interp(length, [0,260],[400,150])
        volPer = np.interp(length, [0,260],[0,100])


        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img,(cx,cy), 15,(0,255,255), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %',(40,450), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)

    #calculate fps
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime
    
    cv2.putText(img, f'FPS: {int(fps)} ',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)