### Problem description
Develop a machine learning system that, based on a list of statically imported libraries of an $\texttt{exe}$ file, predicts whether this file is malicious.
To complete the task, three samples are provided: training, validation and testing. Selections are presented in the form of $\texttt{tsv}$ files with three columns:
1) $\texttt{isvirus}$ – whether the file is malicious: 1 = yes, 0 = no; 
2) $\texttt{filename}$ – file name for review; 
3) $\texttt{libs}$ – comma-separated list of libraries statically imported by this file (we used the LIEF library to obtain the list).
### Model
TF-IDF approach and Naïve Bayes Classification were chosen taking into account the imbalance between 2 classes.
### Some EDA
<img width="600" src="https://github.com/verse-chorus/infotecs_ml/assets/61665391/fcbf29f9-7812-425f-8699-561d10477ecc">

### Metrics on validation
1) True positive: 666
2) False positive: 116
3) False negative: 134
4) True negative: 284
5) Accuracy: 0.7917
6) Precision: 0.8517
7) Recall: 0.8325
8) F1: 0.8420
