# import cv2 
# import numpy as np

# recog=cv2.face.LBPHFaceRecognizer_create()

# recog.read("facemodel.yml")

# test_image=f"yash.jpg"

# test_data=cv2.imread(test_image,0)

# id,confi=recog.predict(test_data)

# print(f'id : {id} , confi : {confi}')   

# cv2.imshow("Result",test_data)
# cv2.waitKey()


import cv2

cam=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recog=cv2.face.LBPHFaceRecognizer_create()
recog.read("facemodel.yml")

names={0:"harshitha",1:"ananya",2:"milana",4:"Poornimi"}

while True:
    flag,frame=cam.read()

    if flag:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,1.1,5)

        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            roi=gray[y:y+h,x:x+w]
            roi=cv2.resize(roi,(300,300))

            id,confi=recog.predict(roi)

            name="unknown"
            if confi < 50:
                name=names[id]

            cv2.putText(frame,name,(x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Face Recognition",frame)

        if cv2.waitKey(1)==ord('q'):
            break

cam.release()
cv2.destroyAllWindows()