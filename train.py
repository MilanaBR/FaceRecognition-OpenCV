from os import listdir
import cv2
import numpy as np


recog=cv2.face.LBPHFaceRecognizer_create()
root_dir="./dataset"

features=[] # x i.e input
labels=[]  # y i.e expected output

# y=f(x) where f is your ML algorithm ( LBPH )

i=0

for folder in listdir(root_dir):
    #print(folder)

    folder_path=f"{root_dir}/{folder}"

    print(f"--------(folder_path)--------")

    for file in listdir(folder_path):
        print(file)

        file_path=f"{folder_path}/{file}"

        img=cv2.imread(file_path,0)

        features.append(img)
        labels.append(i)
    i=i+1

print(f"Features ares {features}")
print(f"Labels are {labels}")

recog.train(features,np.array( labels ))

#above line trains the model using LBPH algorithm

recog.save("facemodel.yml")

        # cv2.imshow("demo",img)
        # cv2.waitKey()
        # print(img)

