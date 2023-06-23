import cv2
import mediapipe as mp
import time
import numpy as np
import math

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volrange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

min_vol = volrange[0]
max_vol = volrange[1]

cam = cv2.VideoCapture(0)

mhand = mp.solutions.hands
hands = mhand.Hands()
mpdraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
x1,x2,y1,y2 = 0,0,0,0
vol = 0
volbar = 0
volper = 0
pTime = 0
cTime = 0


while True:
    success, img = cam.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lists = [id,cx,cy]
                if id == 4 or id == 8:
                    if id == 4:
                        x1, y1 = lists[1], lists[2]
                        centrex = (x1 + x2) // 2
                        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    if id == 8:
                        x2, y2 = lists[1], lists[2]
                        centrey = (y1 + y2) // 2
                        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.circle(img, (centrex, centrey), 10, (0, 255, 0), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [50, 300], [min_vol, max_vol])
                volbar = np.interp(length, [50, 300], [400,150])
                volper = np.interp(length, [50, 300], [0,100])
                volume.SetMasterVolumeLevel(vol,None)

            cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
            cv2.rectangle(img,(50,int(volbar)),(85,400),(0,255,0),cv2.FILLED)
            cv2.putText(img,f"{int(volper)} %",(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)


        mpdraw.draw_landmarks(img, handlms, mhand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
