## Prepare annotations
import pickle
import os
import cv2

root = "../data/Img_celebA/"
# input_pkl = "../data/annotations/CelebA_test.pkl"
# output_pkl = "../data/annotations/CelebA_test_.pkl"

input_pkl = "../data/annotations/CelebA_train.pkl"
output_pkl = "../data/annotations/CelebA_train_.pkl"

# input_pkl = "../data/annotations/CelebA_val.pkl"
# output_pkl = "../data/annotations/CelebA_val_.pkl"


with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

filenames = data.keys()

results = {}

for filename in filenames:
    if os.path.exists(root + filename):
      results[filename] = data[filename]
    else:
      print(root + filename)
with open(output_pkl, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("DONe")