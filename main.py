import cv2
import cvzone
from ultralytics import YOLO
import time
import math
cap = cv2.VideoCapture(0)
cap.set(3,6400)
cap.set(4,480)

model = YOLO('../models/nversion300.pt')

classNames=["fake","real"]

prev_frame_time=0
new_frame_time=0

while True:
    new_frame_time=time.time()
    success, img = cap.read()
    results=model(img,stream=True,verbose=False)
    for r in results:
        boxes= r.boxes
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            conf=math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    print(fps)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





