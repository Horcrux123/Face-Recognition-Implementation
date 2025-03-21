import cv2
import pathlib
cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/ "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
camera = cv2.VideoCapture(0)
#logic
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 255, 55), 2)

    cv2.imshow("pottan",frame)
    if cv2.waitKey(1) == ord("f"):
        break
camera.release()
cv2.destroyAllWindows()
