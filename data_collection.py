import cv2
import random

cam=cv2.VideoCapture(0)

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    flag,frame=cam.read()

    if flag:
    

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


        faces=cascade.detectMultiScale(gray,1.1,5)

        print(faces)

        for x,y,w,h in faces:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


        cv2.imshow("Mycam",frame)
        k=cv2.waitKey(1)

        if k==ord('q'):
            break 

        if k==ord('s'):

            if len(faces) > 0:
            
               roi=gray[y:y+h,x:x+w] 

               roi=cv2.resize(roi,(300,300))  

               num=random.randint(1,100)

               file_name=f"./dataset/3/person{num}.jpg"
               cv2.imwrite(file_name,roi)  

cam.release()