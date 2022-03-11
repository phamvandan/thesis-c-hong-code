import pickle

root = "../data/Helen_test/HR/"

with open("../data/annotations/CelebA_test.pkl", 'rb') as f:
    x = pickle.load(f)

print(len(x.keys()))