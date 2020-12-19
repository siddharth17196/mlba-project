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
    l = [float(i)/total_dipeptides for i in l]
    return l

def c_term(a, length):
    l=[0]*length
    for i in range(min(length, len(a))):
        l[length-i-1] = seq[a[len(a)-1-i]]
    return l

def n_term(a, length):
    l=[0]*length
    for i in range(min(length, len(a))):
        l[i] = seq[a[i]]
    return l


features = []
for i in tqdm(range(len(data['sequence']))):
    f1 = aac(data['sequence'][i])
    f2 = dpc(data['sequence'][i])
    f = f2
    f.append(data['label'][i])
    features.append(f)

with open("dpc_data.pkl", "wb") as fil:
    pickle.dump(features, fil)