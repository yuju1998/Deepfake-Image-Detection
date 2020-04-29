import pickle

man = ["DF", "NT", "F2F", "FS"]

man_comb = []
for man_type in man:
    with open("tmp/%s_tmp.pkl"%man_type, 'rb') as f:
        a = pickle.load(f)
        man_comb.append(a)
    with open("tmp/original_tmp.pkl", 'rb') as f:
        b = pickle.load(f)
    
    c = {}
    for i in a:
        c[i] = a[i]
    for i in b:
        c[i] = b[i]

    with open('c23_%s.pkl'%man_type, 'wb') as f:
        print(len(c))
        pickle.dump(c, f)

'''
with open("original_tmp.pkl", 'rb') as f:
    b = pickle.load(f)
    man_comb.append(b)

all_comb = {}
for i in man_comb:
    for vid in i:
        all_comb[vid] = i[vid]

with open('c23_all.pkl', 'wb') as f:
    print(len(all_comb))
    pickle.dump(all_comb, f)
'''

'''
b = {}
with open('%s.pkl'%man, 'rb') as f:
    a = pickle.load(f)

    for i in a:
        b["original_%s"%i] = a[i]

with open('%s_tmp.pkl'%man, 'wb') as f:
    pickle.dump(b, f)

b = {}
for man_type in man:
    with open('c23_%s.pkl'%man_type, 'rb') as f:
        a = pickle.load(f)

    for i in a:
        b[i] = a[i]

with open('c23_all.pkl', 'wb') as f:
    pickle.dump(b, f)
'''

#with open('c23_all.pkl', 'rb') as f:
#    a = pickle.load(f)

#for i in a:
#    print(i)
