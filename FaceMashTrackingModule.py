import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
faces = mp.solutions.face_mesh
face = faces.FaceMesh()
mpdraw = mp.solutions.drawing_utils
mpdrawspecs = mpdraw.DrawingSpec(thickness=1,circle_radius=1,color=(0,255,0))


while True:
    sucess, img = vid.read()
    cvtimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = face.process(cvtimg)

    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            mpdraw.draw_landmarks(img,facelms,faces.FACEMESH_FACE_OVAL,landmark_drawing_spec=mpdrawspecs)

    cv2.imshow("Face Mash Detection", img)
    cv2.waitKey(10)