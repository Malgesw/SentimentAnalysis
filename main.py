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


def tag_words(file_path, list1, list2, tag, old_directory, new_directory):
    with open(file_path, 'r') as f:
        words = f.read().split()

    can_tag = True

    for i in range(len(words)):
        if words[i] in list1:
            i = i + 1
            while can_tag and i < len(words):
                if words[i] not in list2:
                    words[i] = tag + words[i]
                    i = i+1
                else:
                    can_tag = False

    new_file_path = file_path.replace('.txt', '_tagged.txt')
    new_file_path = new_file_path.replace(old_directory, new_directory)
    with open(new_file_path, 'w') as f:
        f.write(' '.join(words))

    return new_file_path


negation_words = {"won't", "wouldn't", "shouldn't", "couldn't", "not", "isn't", "aren't", "haven't",
                  "don't", "doesn't", "wasn't", "weren't", "didn't", "can't", "cannot", "mustn't"}
punctuation_marks = {".", ";", "?", "!", "]", ")", "[", "(", "/", ",", ":"}

pos_dir_path = "C:/Users/nicco/Desktop/PyCharmProjects/AIProjects/SentimentAnalysis/dataset/mix20_rand700_tokens/tokens/pos"
neg_dir_path = "C:/Users/nicco/Desktop/PyCharmProjects/AIProjects/SentimentAnalysis/dataset/mix20_rand700_tokens/tokens/neg"

for file in os.listdir(neg_dir_path):
    filePath = os.path.join(neg_dir_path, file)
    tag_words(filePath, negation_words, punctuation_marks, 'NOT_', "dataset/mix20_rand700_tokens/tokens/neg", 'prova 2')

