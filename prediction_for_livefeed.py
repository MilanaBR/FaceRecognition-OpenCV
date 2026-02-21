import cv2

cam=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recog=cv2.face.LBPHFaceRecognizer_create()

recog.read("facemodel.yml")

names={0:"harshitha",1:"ananya",2:"milana",3:"Poornimi"}

while True:


    flag,frame=cam.read()

    if flag:
        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=cascade.detectMultiScale(gray,1.1,5)


        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+w),(255,0,0),2)


        if len(faces) > 0 :

            x,y,w,h=faces[0]

            roi=gray[y:y+h,x:x+w]

            roi=cv2.resize(roi,(300,300))

            id,confi=recog.predict(roi)

            print(f" id : {id} , confi : {confi} {names[id]}")

            name="unkown"
            if confi < 50 :
                name=names[id]

            cv2.putText(frame,
            name,            # Text
            (50, 50),                  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,                         # Font scale
            (0, 255, 0),               # Color (Green - BGR)
            2,                         # Thickness
            cv2.LINE_AA)               # Line type



        cv2.imshow("Face Recognition",frame)
        k=cv2.waitKey(1)


        if k==ord('q'):
            break

cam.release()