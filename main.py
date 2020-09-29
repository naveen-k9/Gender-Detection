from tensorflow.keras.preprocessing.image import  img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

#to detect a face
import cvlib as cv

#load the model
model = load_model('genderDetection.model')

#open camera
cam  = cv2.VideoCapture(1)

classes = ['man','woman']

while cam.isOpened():

    # frame
    status, frame = cam.read()
    print(status)
    #detect the face in the frame
    face,confidence = cv.detect_face(frame)
    print(face)

    # iterate through detected faces
    for i, f in enumerate(face):

        #corner points of rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        cropFace = np.copy(frame[startY:endY,startX:endX])

        if (cropFace.shape[0]) < 10 or (cropFace.shape[1]) < 10:
            continue

        # preprocessing for model
        cropFace = cv2.resize(cropFace, (96,96))
        cropFace = cropFace.astype("float") / 255.0
        cropFace = img_to_array(cropFace)
        cropFace = np.expand_dims(cropFace, axis=0)

        #apply model
        con = model.predict(cropFace)[0] # returns 2D matrix

        # get label with max accuracy
        i = np.argmax(con)
        label = classes[i]

        label = "{}: {:.2f}%".format(label, con[i]*100)

        Y = startY - 10 if startY - 10 > 10 else startY * 10

        #put label and confidence on image
        cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    #display
    cv2.imshow("Gender Detection", frame)

    #press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release
cam.release()
cv2.destroyAllWindows()




