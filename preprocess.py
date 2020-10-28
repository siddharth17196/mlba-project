import pandas as pd
from tqdm import tqdm
import pickle

data = pd.read_csv("./dataset.csv")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
seq = {}
for i in range(len(amino_acids)):
    seq[amino_acids[i]] = i

dipep_seq = {}
k = 0
for i in range(len(amino_acids)):
    for j in range(len(amino_acids)):
        dipep_seq[amino_acids[i]+amino_acids[j]] = k
        k += 1

def aac(a):
    l = [0]*20
    la = len(a)
    for i in range(la):
        c = a.count(a[i])
        l[seq[a[i]]] = c/la
    return l

def dpc(a):
    l = [0]*400
    la = len(a)
    total_dipeptides = la**2
    for k in range(3):
        i = 0
        j = i+k
        while j < la:
            l[dipep_seq[a[i]+a[j]]] += 1
            i+=1
            j+=1
    # l = l/total_dipeptides
    l = [float(i)/total_dipeptides for i in l]
    return l

features = []
for i in tqdm(range(len(data['sequence']))):
    f1 = aac(data['sequence'][i])
    f2 = dpc(data['sequence'][i])
    f = f1 + f2
    f.append(data['label'][i])
    features.append(f)

with open("data.pkl", "wb") as fil:
    pickle.dump(features, fil)