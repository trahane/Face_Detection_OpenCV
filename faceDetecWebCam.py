import cv2
import numpy

def checkFace(xmlCascadeFile):
    #0 for inbult cam & 1 for external cam
    videoFeed = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(xmlCascadeFile)

    while True:
        # Capture frame-by-frame
        ret, frame = videoFeed.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



        # Display the resulting frame

        #for resizeable window
        #cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # count = 0
        # for each in faces:
        #     count += 1
        #     print count
            # When everything is done, release the capture
    videoFeed.release()
    cv2.destroyAllWindows()

checkFace('haarcascade_frontalface_default.xml')

