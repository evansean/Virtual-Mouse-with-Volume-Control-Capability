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
area = 0
while True:
    sucess,img = cap.read()

    #Find Hand
    img = detector.findHands(img)
    lmList,bbox = detector.findPos(img,draw=True)
    # print(bbox)
    if len(lmList)!= 0:

        # filter based on size of the hand
        area = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))//100
        # print(area)
        if 120 < area < 1500:

            # find distance between index and thumb
            length, img, lineInfo = detector.findDistance(4,8,img)
            print(length)

            # convert length to volume
            # Hand range 15-200
            #Vol range -96.0 - 0.0
            #calibrate range in the future
            volBar = np.interp(length, [30,260],[400,150])
            volPer = np.interp(length, [30,260],[0,100])
            # volume.SetMasterVolumeLevel(vol, None)
            # reduce resolution to make it smoother
            smooth=5
            volPer=smooth*round(volPer/smooth)
            # check fingers up (if pinky is up/down, set volume)
            fingers = detector.fingersUp()
            if fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100,None)
                cv2.circle(img,(lineInfo[4],lineInfo[5]), 15,(0,255,255), cv2.FILLED)

            

           

    # Drawings
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