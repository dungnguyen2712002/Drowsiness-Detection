import cv2
import dlib
from imutils import face_utils
import pandas as pd
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from preprocess_data import filter_col


filename = 'finalized_model.sav'
frame_to_keep = 150
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
path_pred_68 = os.path.join('facial-landmarks','shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_pred_68)
loaded_model = pickle.load(open(filename, 'rb'))


landmark_pts_count = 68
features = ['mood']

for i in range(landmark_pts_count):
    features.append('px_'+str(i+1))
    features.append('py_'+str(i+1))

def model(landmarks):
    landmarks = np.array(landmarks, dtype=np.float64)
    df = pd.DataFrame(landmarks, columns=features)
    df.replace(-1, np.NaN, inplace=True)
    df.interpolate(inplace=True, limit_direction='both')
    df = filter_col(df)
    df = df.drop('mood', axis=1)
    X = df.to_numpy().flatten().reshape(1, -1)
    ypred = loaded_model.predict(X)
    return ypred[0].astype(str)

def live():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    data = np.full((100, 137), -1).tolist()
    result = []
    state = 'Caliberating...'
    start = 0
    while True:
        # Getting out image by webcam
        _, image = cap.read()
        end = time.time()
        cv2.putText(image, state, bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)

        # if len(data) < frame_to_keep-1:
        #     cv2.putText(image, 'Caliberating...', bottomLeftCornerOfText,
        #                 font, fontScale, fontColor, lineType)

        if len(data) > frame_to_keep-1:
            # we only need last 149 frames
            data = data[len(data)-149:]

        interval = end - start

        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(image, 0)

        # if no faces are found
        if len(rects) == 0:
            data.append(['-1' for x in range(1+landmark_pts_count*2)])
            # cv2.putText(image, 'Caliberating...', bottomLeftCornerOfText,
            #             font, fontScale, fontColor, lineType)
        elif len(rects) >= 1 :
            rect = rects[0]

            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # we only add frames with interval of 100ms
            if interval >= 0.1:
                frame_data = [-1] + shape.flatten().squeeze().tolist()
                data.append(frame_data)
                if len(data) == frame_to_keep:
                    mood = model(data)
<<<<<<< HEAD
                    print(mood)
                    if mood == '0':
                        state = 'Alert'
                    # elif mood == '5':
                    #     state = 'Normal'
=======
                    if mood == '0':
                        state = 'Alert'
                    elif mood == '5':
                        state = 'Normal'
>>>>>>> bb908a6b9eb11718456abc1af768b7ce6614d460
                    else:
                        state = 'Drowsy'
                    # cv2.putText(image, state, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                    result.append(mood)
                    data = data[1:]
                    start = end

            # Draw on our image, all the finded cordinate points (x,y)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("Output", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    return data, result

if __name__ == '__main__':
    live()