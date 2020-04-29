import pickle
b = {}

man = "original"
with open('%s.pkl'%man, 'rb') as f:
    a = pickle.load(f)

    for i in a:
        b["original_%s"%i] = a[i]

with open('%s_tmp.pkl'%man, 'wb') as f:
    pickle.dump(b, f)

