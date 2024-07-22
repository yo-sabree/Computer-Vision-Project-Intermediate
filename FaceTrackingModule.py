import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
cTime = 0
pTime = 0

mpface = mp.solutions.face_detection
face = mpface.FaceDetection()
mpdraw = mp.solutions.drawing_utils


while True:
    success, img = vid.read()
    imgcvt = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = face.process(imgcvt)
    if result.detections:
        for detection in result.detections:
            #print(detection)
            temp = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            box = int(temp.xmin*w) , int(temp.ymin*h) , int(temp.width*w),int(temp.height*h)
            #mpdraw.draw_detection(img,detection)
            cv2.rectangle(img,box,(234,132,111),2)
            cv2.putText(img, str(int(detection.score[0]*100))+"%", (box[0],box[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Face Tracking Module", img)
    cv2.waitKey(1)