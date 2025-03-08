from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import  cv2

cap = cv2.VideoCapture(0)


detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)


while True:

    success, img = cap.read()


    img, bboxs = detector.findFaces(img, draw=True)


    if bboxs:
            # Loop through each bounding box
        for bbox in bboxs:

            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

                # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            #cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            #cvzone.cornerRect(img, (x, y, w, h))

        # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)
