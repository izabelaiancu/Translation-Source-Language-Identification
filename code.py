import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import re
import os


def files_in_folder(mypath):
    return sorted([os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))])





# incarca datele
data = []
# schimba numele lui dir_path
dir_path = 'trainData/trainData/'
folder_path = dir_path + 'trainExamples'
labels = np.loadtxt(dir_path + 'labels_train.txt')
for fis in files_in_folder(folder_path):
    with open(fis, 'r', encoding='utf-8') as fin:
        text = fin.read()
        linie = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text).split()
        listToStr = ' '.join(map(str, linie))  # deoarece tfidf lucreaza cu stringuri
        data.append(listToStr)


print(data[0])

# Crearea dictionarului text : label
category_to_id = dict()

for n in range(0, len(data)):
    linie = data[n]
    label = labels[n]
    category_to_id.update({linie: label})

print('lungimea dict este: %d' % len(category_to_id))

# TEXT PROCCESING

tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=20000, min_df=2)
features1 = tfidf.fit_transform(category_to_id.keys())

print("Each of the %d texts is represented by %d features (TF-IDF score of unigrams and bigrams)" % (features1.shape))



# Impartirea setului de date in date de antrenare si date de testare

X_train, X_test, y_train, y_test = train_test_split(features1, labels, test_size=0.25, random_state=0)



# Cautarea celui mai bun C pentru cea mai buna acuratete
acurateti = []
ceuri = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100,1000]
for c in ceuri:
    model = LinearSVC(C=c)
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    acurateti.append(accuracy_score(y_test, p))
acurateti = np.array(acurateti)
c_max = acurateti.max()
c_maibun = ceuri[acurateti.argmax()]
print('Lista accuratetilor obtinute: ', acurateti)
print('Acuratetea maxima este: %f' % c_max)
print('Best C este: ', c_maibun)

model_svc = LinearSVC(C = c_maibun)
model_svc.fit(X_train, y_train)
p_svc = model_svc.predict(X_test)


# K Fold cross validation
kfold = model_selection.KFold(n_splits=10, random_state=100)
model_kfold = LinearSVC(C = c_maibun)
results_kfold = model_selection.cross_val_score(model_kfold, features1, labels, cv=kfold)
print("Acuratete Kfold cross validation: %.2f%%" % (results_kfold.mean()*100.0))

# Accesarea si prelucrarea datelor de testare
date_de_test = 'testData-public/testData-public/'

data_test = dict()
for fis in sorted(files_in_folder(date_de_test)):
    with open(fis, 'r', encoding='utf-8') as fin:
        text = fin.read()
        nume_fisier = os.path.basename(fis)
        idtx = nume_fisier.replace('.txt', '')
        linie = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text).split()
        listToStr = ' '.join(map(str, linie))  # deoarece tfidf lucreaza cu stringuri
        data_test[idtx] = listToStr


data_test_tfidf = tfidf.transform(data_test.values()) # nu mai e nevoie de fit_transform deoarece este already fitted

# Clasificatorul cel mai accurate pe datele de testare nevazute
clf = LinearSVC(C = c_maibun)
start_time = time.time()
clf.fit(X_train, y_train)
print("Training time --%s seconds --" % (time.time() - start_time))
predictii = clf.predict(data_test_tfidf)


# Confusion matrix
nr_clase = len(np.unique(labels))
confusion_matrix = np.zeros((nr_clase, nr_clase), 'int8')
for adevar, predictie in zip(y_train,p_svc):
    adevar = int(adevar)
    predictie = int(predictie)
    confusion_matrix[adevar, predictie] += 1
print("Confusion Matrix: ")
print(confusion_matrix)


# Confusion matrix fancy plot
import seaborn as sns

fig, ax = plt.subplots(figsize = (nr_clase, nr_clase))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='d',
            xticklabels=np.unique(labels),
            yticklabels=np.unique(labels))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=10)
plt.show()





'''
    
with open('c mai bun.csv', 'w') as fout:
    fout.write("Id,Prediction\n")
    iduri = data_test.keys()
    for idtx, p in zip(iduri, predictii):
        fout.write(idtx + ',{}\n'.format(int(p)))


'''