import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mhand = mp.solutions.hands
hands = mhand.Hands()
mpdraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                print(id,cx,cy)
                if(id==4):
                    cv2.circle(img,(cx,cy),25,(255,0,0),cv2.FILLED)
        mpdraw.draw_landmarks(img,handlms,mhand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)




