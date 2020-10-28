from Bio import SeqIO
import pandas as pd

sequence = []
label = []
for record in SeqIO.parse("positive.fasta", "fasta"):
    sequence.append(record.seq)
    label.append(1)

for record in SeqIO.parse("negative.fasta", "fasta"):
    sequence.append(record.seq)
    label.append(0)

df = pd.DataFrame({
        "sequence": sequence,
        "label": label
    })
df.to_csv("dataset.csv",index=False)
