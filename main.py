"""
python implementation of the experiments described in "Thumbs up? Sentiment Classification using Machine Learning
Techniques" (B. Pang, L. Lee, S. Vaithyanathan (2002))

------------------------------------------
Type of features:

#1: 16165 unigrams by frequency (first 16165 unigrams which occured more than four times in the corpus)
#2: 16165 unigrams by presence (first 16165 unigrams which occured the most, but binarizing feature vectors)
#3: 16165 unigrams + 16165 bigrams by presence (first 16165 unigrams + bigrams which occured the most, " ")
#4: 16165 bigrams by presence (first 16165 bigrams wich occured the most, " ")
#5: unigrams + POS (MISSING DATA, QTAG ALGORITHM NOT FOUND)
#6: adjectives (MISSING DATA, QTAG ALGORITHM NOT FOUND)
#7: 2633 unigrams by presence (first 2633 unigrams which occured the most, " ")
#8: 22430 unigrams + position (first 22430 unigrams which occurred the most, " " and tagging every unigram in respect to
its position in the document (first quarter, last quarter, middle half))

------------------------------------------
3-Fold Cross Validation criteria:

for each class of reviews (positive "pos" and negative "neg"):
-fold 1: files tagged cv000 through cv232, in numerical order
-fold 2: files tagged cv233 through cv465, in numerical order
-fold 3: files tagged cv466 through cv699, in numerical order

------------------------------------------
Datasets used (from https://www.cs.cornell.edu/people/pabo/movie-review-data/):

-polarity dataset v1.0 (mix20_rand700_tokens_cleaned) (maybe)
-polarity dataset v0.9 (mix20_rand700_tokens)

------------------------------------------
Before training the model and fitting the data, documents were modified such that, between every negation word and a
punctuation mark, all words contained the tag "NOT_" (Example: I won't do it. -> I won't NOT_do NOT_it.)

Negation words = {won't, wouldn't, shouldn't, couldn't, not, isn't, aren't, haven't, don't, doesn't, wasn't, weren't,
can't, cannot}
Punctuation marks = {".", ";", "?", "!", "]", ")", "[", "(", "/", ",", ":", "'"}

"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import glob
import os
import random


def tag_words(file_path, list1, list2, tag, old_directory, new_directory):
    with open(file_path, 'r') as f:
        words = f.read().split()

    can_tag = True

    for j in range(len(words)):
        if words[j] in list1:
            j = j + 1
            while can_tag and j < len(words):
                if words[j] not in list2:
                    words[j] = tag + words[j]
                    j = j+1
                else:
                    can_tag = False

    new_file_path = file_path.replace('.txt', '_tagged.txt')
    new_file_path = new_file_path.replace(old_directory, new_directory)
    with open(new_file_path, 'w') as f:
        f.write(' '.join(words))

    return new_file_path


def read_files(file_paths):
    reviews = []
    for fp in file_paths:
        with open(fp, 'r') as f:
            reviews.append(f.read())
    return reviews

# SETTING UP THE DATA

negation_words = {"won't", "wouldn't", "shouldn't", "couldn't", "not", "isn't", "aren't", "haven't",
                  "don't", "doesn't", "wasn't", "weren't", "didn't", "can't", "cannot", "mustn't"}
punctuation_marks = {".", ";", "?", "!", "]", ")", "[", "(", "/", ",", ":", "..."}

pos_dir_path = "C:/Users/nicco/Desktop/PyCharmProjects/AIProjects/SentimentAnalysis/dataset/mix20_rand700_tokens/tokens/pos"
neg_dir_path = "C:/Users/nicco/Desktop/PyCharmProjects/AIProjects/SentimentAnalysis/dataset/mix20_rand700_tokens/tokens/neg"

files = os.listdir(pos_dir_path.replace("dataset/mix20_rand700_tokens/tokens/pos", "pos_tagged"))

if not files:
    for file in os.listdir(pos_dir_path):
        f_path = os.path.join(pos_dir_path, file)
        tag_words(f_path, negation_words, punctuation_marks, "NOT_", "dataset/mix20_rand700_tokens/tokens/pos",
                  "pos_tagged")
    pos_dir_path = pos_dir_path.replace("dataset/mix20_rand700_tokens/tokens/pos", "pos_tagged")
else:
    print("pos dataset already set up")

files = os.listdir(neg_dir_path.replace("dataset/mix20_rand700_tokens/tokens/neg", "neg_tagged"))

if not files:
    for file in os.listdir(neg_dir_path):
        f_path = os.path.join(neg_dir_path, file)
        tag_words(f_path, negation_words, punctuation_marks, "NOT_", "dataset/mix20_rand700_tokens/tokens/neg",
                  "neg_tagged")
    neg_dir_path = neg_dir_path.replace("dataset/mix20_rand700_tokens/tokens/neg", "neg_tagged")
else:
    print("neg dataset already set up")

# THREE FOLD CROSS VALIDATION

pos_reviews = read_files(glob.glob(pos_dir_path.replace("dataset/mix20_rand700_tokens/tokens/pos", "pos_tagged")+"/*"))
neg_reviews = read_files(glob.glob(neg_dir_path.replace("dataset/mix20_rand700_tokens/tokens/neg", "neg_tagged")+"/*"))

folds = []
labels = []  # 1 -> positive review | 0 -> negative review
fold_1 = pos_reviews[:233]+neg_reviews[:233]
labels.append([1]*233+[0]*233)
folds.append(fold_1)
fold_2 = pos_reviews[233:466]+neg_reviews[233:466]
labels.append([1]*233+[0]*233)
folds.append(fold_2)
fold_3 = pos_reviews[466:700]+neg_reviews[466:700]
labels.append([1]*234+[0]*234)
folds.append(fold_3)

fold_numbers = [1, 2, 3]
remaining_numbers = [1, 2, 3]
c = 0
scores = []

while c < 3:
    k = random.choice(remaining_numbers)
    remaining_numbers.remove(k)
    test_fold = folds[k-1]
    test_labels = labels[k-1]
    train_fold = []
    train_labels = []
    for i in fold_numbers:
        if i != k:
            train_fold.extend(folds[i-1])
            train_labels.extend(labels[i-1])
    cv = CountVectorizer()
    train_vectors = cv.fit_transform(train_fold)
    test_vectors = cv.transform(test_fold)
    clf = MultinomialNB()
    clf.fit(train_vectors, train_labels)  # training phase
    pred = clf.predict(test_vectors)
    accuracy = metrics.accuracy_score(test_labels, pred)
    scores.append(accuracy)
    c = c + 1  # repeat and try with a different test fold

result = round(sum(scores)/c, 3)*100
print("accuracy = "+str(result)+"%")
