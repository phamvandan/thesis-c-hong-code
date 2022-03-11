import pickle
import cv2
import numpy as np

def get_new_landmarks(landmarks):
    new_landmarks = []
    for index in [7,9,11,18,20,22,23,25,27,31,32,34,36]:
        new_landmarks.append(landmarks[index-1])

    points = []
    for index in [37,38,39,40,41,42]:
        points.append(landmarks[index-1])
    new_lm = np.mean(np.array(points), axis=0)
    new_landmarks.append((new_lm[0], new_lm[1]))

    points = []
    for index in [43,44,45,46,47,48]:
        points.append(landmarks[index-1])
    new_lm = np.mean(np.array(points), axis=0)
    new_landmarks.append((new_lm[0], new_lm[1]))

    for index in [49,52,55,58]:
        new_landmarks.append(landmarks[index-1])
    return new_landmarks

root = "../data/Helen_test/HR/"

with open("../data/annotations/Helen_test.pkl", 'rb') as f:
    x = pickle.load(f)

key = list(x.keys())[0]
landmarks = list(x.values())[0]
img_path = root + key

img = cv2.imread(img_path)
ori = img.copy()

new_landmarks = get_new_landmarks(landmarks)
print(len(new_landmarks))
for index, (x,y) in enumerate(new_landmarks):
    print(index)
    cv2.circle(img, (int(x),int(y)), radius=2, color=(0,0,255), thickness=-1)
    cv2.imshow("lm", img)
    cv2.waitKey(0)

# flipHorizontal = cv2.flip(ori, 1)
# h,w = flipHorizontal.shape[:2]
# for index, (x,y) in enumerate(landmarks):
#     print(index)
#     cv2.circle(flipHorizontal, (int(w - x - 1),int(y)), radius=2, color=(0,0,255), thickness=-1)
#     cv2.imshow("flip", flipHorizontal)

cv2.imwrite('img.jpg', ori)
cv2.imshow("original image", ori)
cv2.waitKey(0)

# 0-16 bao quanh ham 0 1 2
# 17-21 long may trai 3 4 5
# 22-26 long may phai 6 7 8
# 27-30 song mui 9
# 31-35 vien mui 10 11 12
# 36-41 mat trai 13
# 42-47 mat phai 14
# 48-53 moi tren 15 16
# 54-59 moi duoi 17 18
# 60-67 giua mieng
