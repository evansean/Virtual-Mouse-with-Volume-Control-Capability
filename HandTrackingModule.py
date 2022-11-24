import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict


class handDetection():
    def __init__(self, mode = False, maxHands = 1 ,modelComplexity=1, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.modelComplexity = modelComplexity
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity, self.detectionConfidence,self.trackConfidence)
        # Use MediaPipe method to draw landmarks
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20] #tip id of thumb, index, middle, ring, pinky finger respectively


    def findHands(self,img, draw = True):
        # OpenCV captures and processes images in BGR, while MediaPipe does so in RGB hence we need to convert
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Store the results from the process method from MediaPipe
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  
        return img        

               
    
    def findPos(self,img, handNo=0, draw = True):
        # list of Landmarks detected
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    height,width, channel = img.shape
                    # convert xy coordinates into pixels by multiplying it with the height and width of img center x y
                    cx,cy = int(lm.x*width), int(lm.y*height)
                    # print(id,cx,cy)
                    self.lmList.append([id,cx,cy])
                    
                    if draw:
                        cv2.circle(img, (cx,cy), 5, (255,255,0), cv2.FILLED)
        return self.lmList

    def handPresence(self,img):
        if self.results.multi_hand_landmarks:
            # Both Hands are present in image(frame)
            if len(self.results.multi_handedness) == 2:
                return 2
            else:
                for i in self.results.multi_handedness:
               
                # Return whether it is Right or Left Hand
                    label = MessageToDict(i)[
                        'classification'][0]['label']
                    if label == "Left":
                        return 1
                    if label == "Right":
                        return 0
                    return -1

    def fingersUp(self):
        #detect which hand is up
        handPres=-1
        if self.results.multi_hand_landmarks:
            # Both Hands are present in image(frame)
            for i in self.results.multi_handedness:
               
                # Return whether it is Right or Left Hand
                    label = MessageToDict(i)[
                        'classification'][0]['label']
                    if label == "Left":
                        handPres= 1
                    if label == "Right":
                        handPres= 0
         # In opencv, the image orientation has higher positions marked as lower values
        fingers=[]

        #for thumbs, check if x coordinate is to the left/right of of -2 landmarks of thumbtip
        #check if left or right hand, adjust thumb check accordingly
        #if left hand
        if handPres == 0:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
                    fingers.append(1)
            else:
                    fingers.append(0)
        #if right hand
        elif handPres == 1:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
                    fingers.append(1)
            else:
                    fingers.append(0)

        #for fingers, check if y coordinate of finger tip is lower than -2 landmarks of fingertip
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    prevTime = 0
    curTime = 0
    # Capture frame from default webcam 0
    cap = cv2.VideoCapture(0)
    detector = handDetection()
    while True:
        # Store frame as an variable
        success, img = cap.read()
        img = detector.findHands(img,True)
        lmList = detector.findPos(img)
        if len(lmList) != 0:
            print(lmList[4])


        #Calculate FPS
        curTime = time.time()
        fps = 1/(curTime-prevTime)
        prevTime = curTime

        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,0,255), 3)
        cv2.imshow("testesss", img)
        cv2.waitKey(1)

if __name__  == "__main__":
        main()