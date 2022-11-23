import cv2
import mediapipe as mp
import time

class handDetection():
    def __init__(self, mode = False, maxHands = 2 ,modelComplexity=1, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.modelComplexity = modelComplexity
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity, self.detectionConfidence,self.trackConfidence)
        # Use MediaPipe method to draw landmarks
        self.mpDraw = mp.solutions.drawing_utils

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
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    height,width, channel = img.shape
                    # convert xy coordinates into pixels by multiplying it with the height and width of img center x y
                    cx,cy = int(lm.x*width), int(lm.y*height)
                    # print(id,cx,cy)
                    lmList.append([id,cx,cy])
                    
                    if draw:
                        cv2.circle(img, (cx,cy), 5, (255,255,0), cv2.FILLED)
        return lmList

        



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