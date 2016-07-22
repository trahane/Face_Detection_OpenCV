import cv2


def checkFace(harCpath):
    faceCascade = cv2.CascadeClassifier(harCpath)

    #image detection
    imgpath = "5.jpg"
        #read image
    img = cv2.imread(imgpath)
        #CONVERT TO GRAY
    grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #detect faces in img
    faces = faceCascade.detectMultiScale(
        grayImg,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print "Found {0} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", img)
    cv2.waitKey(0)
    print faces
checkFace('haarcascade_frontalface_default.xml')