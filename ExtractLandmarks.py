import cv2
import pandas as pd
from mlxtend.image import extract_face_landmarks
import imutils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
df = pd.read_csv('D:/AI PROJECT/Data_csv/construct.csv', sep=',')

for video in range(1, 61):
    for label in [0, 5, 10]:
        video = cv2.VideoCapture('Data/' + video + '/' + str(label) + '.MOV')
        # print(video.get(cv2.CAP_PROP_FPS))

        count = 0
        while True:
            success, frame = video.read()
            if not success: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            gray = imutils.resize(gray, width=350)

            # Find face
            try:
                faces = face_cascade.detectMultiScale(gray, 1.2, 2)
            except:
                continue

            # Make face bigger
            [x, y, w, h] = faces[-1]
            y0 = int(y - 0.1*h)
            y1 = int(y + h + 0.2*h)
            x0 = int(x - 0.1*w)
            x1 = int(x + w + 0.1*w)
            try:
                face = frame[y0:y1, x0:x1]
            except:
                continue

            # Get landmarks
            try:
                landmarks = extract_face_landmarks(face)

                new_row = {}
                new_row['mood'] = label
                for i in range(len(landmarks)):
                    new_row['px_' + str(i+1)] = landmarks[i][0]
                    new_row['py_' + str(i+1)] = landmarks[i][1]

                add_df = pd.DataFrame(new_row, index=[count])
                df = df.append(add_df, ignore_index=False)

                count += 1
            except:
                continue

df.to_csv('Data/big_data.csv', sep=',', header=True)

# data = pd.read_csv('Data_csv/Fold2_part1/16/0.csv', sep=';')
# data.to_csv('Data_csv/Fold2_part1/16/0.csv', sep=',', header=True)