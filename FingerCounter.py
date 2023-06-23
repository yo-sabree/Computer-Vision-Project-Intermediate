import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
mhand = mp.solutions.hands
hands = mhand.Hands()
mpdraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

fingerlist = [4, 8, 12, 16, 20]
finger = []

while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            lists = []
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lists.append([id, cx, cy])

            if len(lists) != 0:
                finger.clear()
                for id in fingerlist:
                    if id == 4:
                        if lists[0][1] < lists[0 - 1][1]:
                            finger.append(1)
                        else:
                            finger.append(0)
                    else:
                        if lists[id][2] < lists[id - 2][2]:
                            finger.append(1)
                        else:
                            finger.append(0)

                print(finger)
                totalfinger = finger.count(1)



            mpdraw.draw_landmarks(img, handlms, mhand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(1)
