import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "Documents", "python", "buku", "pelatihan", "DATA"
)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):

    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):

        data_aux = []
        X_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))

        if img is None:
            continue  # skip kalau gambar gagal dibaca

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    X_.append(landmark.x)
                    y_.append(landmark.y)

                min_x = min(X_)
                min_y = min(y_)

                for i in range(len(X_)):
                    data_aux.append(X_[i] - min_x)
                    data_aux.append(y_[i] - min_y)

            data.append(data_aux)
            labels.append(dir_)

save_path = os.path.join(os.path.dirname(DATA_DIR), 'data.pickle')

with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data berhasil disimpan!")