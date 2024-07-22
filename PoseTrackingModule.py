import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
mppose = mp.solutions.pose
pose = mppose.Pose()

mpdraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0


while True:
    success, img = vid.read()
    cvtimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    poses = pose.process(cvtimg)
    if poses.pose_landmarks:

        mpdraw.draw_landmarks(img,poses.pose_landmarks,mppose.POSE_CONNECTIONS)
        for id, lm in enumerate(poses.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img,(cx,cy),2,(0,255,0),cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Pose",img)
    cv2.waitKey(10)