# mlba-project

ML for biomedical applications course project

usage: main.py [-h] [-m MODEL]

Course Project

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        svm=1, decision tree=2, random forest=3



```
pip install -r requirements.txt
```

- data.py -> creates a csv of of all proteins and corresponding labels.
- preprocess.py -> create feature vectors using amino acid and dipeptide composition.
- svm.py -> training and prediction.
