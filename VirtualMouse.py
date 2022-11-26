import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def main():
    #-----VARIABLES-------
    widthCam, heightCam = 640,480
    widthScreen,heightScreen = autopy.screen.size() #1280,720
    frameR = 120 #frame reduction
    smooth = 2
    pLocX,pLocY = 0,0
    curLocx,curLocY =0,0
    prevTime=0
    active=0
    devices = AudioUtilities.GetSpeakers() #here to edit main speaker source
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange() #(-96.0, 0.0, 1.5)
    minVol = volRange[0]
    maxVol = volRange[1]
    vol=0
    volBar=400
    volPer=0
    area = 0
    mode = 'deactivation'
    #-----VARIABLES-------
    cap = cv2.VideoCapture(0)
    #3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    #4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    cap.set(3,widthCam)
    cap.set(4,heightCam)
    detector = htm.handDetection(detectionConfidence=0.7)
    while True:
        # Find hand landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmList,bbox = detector.findPos(img)
        fingers=[]
        cv2.rectangle(img,(frameR,frameR), (widthCam-frameR,heightCam-frameR),(255,0,255),2)

    

        if len(lmList) != 0:
            fingers = detector.fingersUp()
            # Get coordinates of index tip
            x1,y1 = lmList[20][1:]

            #--------MODE SELECTION-----------#
            # Deactivate Detection
            # Point middle finger to deactivate detection
            if fingers == [0,0,1,0,0]:
                mode = 'deactivation'

            if mode == 'deactivation':
                active= 1
                # raise pinky only to activate detection
                if fingers == [0,0,0,0,1]:
                    active=0
                    mode = ''
                continue
        
            # Cursor mode: All fingers up
            if fingers == [1,1,1,1,1] and active==0:
                active=1
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
                active=0
            # Clicking mode
            if fingers == [1,0,1,1,1]:
                pyautogui.leftClick()
            if fingers == [1,1,0,1,1]:
                pyautogui.rightClick()
            #Scroll up
            if fingers == [0,1,0,0,0] and active==0:
                active=1
                pyautogui.scroll(50)
                active=0
            #Scroll down
            if fingers == [0,1,1,0,0] and active==0:
                active=1
                pyautogui.scroll(-50)
                active=0
            #Volume control
            if fingers == [0,0,0,0,0] and active==0:
                active=1
                mode = 'volMode'
            
            

            if mode == 'volMode':
                active=1
                area = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))//100
                if 80 < area < 1500:
                    # find distance between index and thumb
                    length, img, lineInfo = detector.findDistance(4,8,img)

                    # convert length to volume
                    # Hand range 15-200
                    #Vol range -96.0 - 0.0
                    #calibrate range in the future
                    volBar = np.interp(length, [20,160],[400,150])
                    volPer = np.interp(length, [20,160],[0,100])
                    # volume.SetMasterVolumeLevel(vol, None)
                    # reduce resolution to make it smoother in terms of steps
                    smooth=5
                    volPer=smooth*round(volPer/smooth)
                    cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
                    cv2.rectangle(img, (50,int(volBar)), (85,400), (0,255,0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %',(40,450), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)
                    if fingers == [1,1,1,1,1]:
                        active=0
                        mode=''
                    if not fingers[4]:
                        volume.SetMasterVolumeLevelScalar(volPer/100,None)
                        # volume.SetMasterVolumeLevel(volPer//100, None)

        #calculate fps
        curTime = time.time()
        fps = 1/(curTime-prevTime)
        prevTime = curTime
        
        # Display
        cv2.putText(img, f'FPS: {int(fps)} ',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 3)
        cv2.imshow("Gesture Controlled Mouse + Volume Controller",img)
        if cv2.waitKey(1) & 0xff == ord('x'):
                break
if __name__ == '__main__':
    main()