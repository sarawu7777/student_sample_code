# Classifiers for yelp and IMBD reviews

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import re
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score



def generate_dict(df, occurence=10001):
    vocab_list = []
    for index, row in df.iterrows():
        sentences = re.sub(r'[^A-Za-z0-9]'," ",df.iloc[index,0]).lower()
        words = sentences.split(" ")
        for w in words:
            vocab_list.append(w)
        # df.iloc[index,0] = sentences.lower()
    word_counts = Counter(vocab_list)
    words = word_counts.most_common(occurence)
    return words


def get_vocablist(counts, DictName):
    dict_out = pd.DataFrame(counts[1:], columns=['word', 'Frequncy'])
    dict_out['ID'] = list(range(10000))
    with open(DictName,"w") as txt:
        for index, row in dict_out.iterrows():
                txt.write(row[0])
                txt.write(" ")
                txt.write(str(row[2]))
                txt.write(" ")
                txt.write(str(row[1]))
                txt.write("\n")
    vocablist = dict_out['word'].T.tolist()
    return vocablist


# extract unique vocabulary set from a given review sentence
def gen_Vocab(reviews):
    sentences = re.sub(r'[^A-Za-z0-9]'," ",reviews).split(" ")
    reviewdat = [sen.lower() for sen in sentences if len(sen) > 0]
    return list(reviewdat)


# Binary bag-of-words representation
# generate the vector, 1 if the dict contains the word, 0 otherwise
def BinBagofwordsVec(inputSet, vocabList):
    Vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            Vec[vocabList.index(word)] = 1
    return Vec


# Frequency bag-of-words representation
# generate the vector, 1 if the dict contains the word, 0 otherwise
def FreqBagofwordsVec(inputSet, vocabList):
    Vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            Vec[vocabList.index(word)] += 1
    s = sum(Vec) + 10000
    Vec = [(V+1)/s for V in Vec]
    return Vec


def reviews_to_numeric(inputSet, vocabList):
    outlist = [vocabList.index(word) for word in inputSet if word in vocabList]
    out = ' '.join(str(e) for e in outlist)
    return out


# Output numeric representation of data
def output_numeric_reviews(df, vocabList, out_name):
    numeric_reviews = df[0].apply(gen_Vocab).apply(reviews_to_numeric, args = (vocabList,))
    numeric_out = pd.concat([numeric_reviews, df[1]], axis = 1)
    with open(out_name,"w") as txt:
        for index, row in numeric_out.iterrows():
            txt.write(row[0])
            txt.write(" ")
            txt.write(str(row[1]))
            txt.write("\n")


def load_feature_mat(df,vectype,vocabList, out = False, OutName = "out.txt"):
    a,_ = df.shape
    X_mat = np.zeros((a, 10000))
    for index, row in df.iterrows():
        inputvocab = gen_Vocab(row[0])
        if(vectype == "binary"):
            vec = BinBagofwordsVec(inputvocab, vocabList)
        else:
            vec = FreqBagofwordsVec(inputvocab, vocabList)
        X_mat[index] = vec
    t = df[1].values
    print("Dimention of design matrix is: ",X_mat.shape)
    Full_mat = np.c_[X_mat, t]
    if(out == True):
        np.savetxt(OutName,X_mat,fmt='%.5f')
    return X_mat, t, Full_mat



def load_data(datatype, procedure):
    train =  pd.read_csv("data/%s-train.txt" %datatype , sep="\t", header=None)
    counts = generate_dict(train, 10001)

    print("loading and processing training data...")
    vocablist = get_vocablist(counts, "data/%s_vocab.txt" %datatype)
    output_numeric_reviews(train, vocablist, "data_processed/%s_train.txt" %datatype)
    train_X, train_y, _ = load_feature_mat(train, procedure, vocablist)
    print("")

    print("loading and processing validation data...")
    valid = pd.read_csv("data/%s-valid.txt" %datatype , sep="\t", header=None)
    output_numeric_reviews(valid, vocablist, "data_processed/%s_valid.txt" %datatype)
    valid_X, valid_y, _ = load_feature_mat(valid, procedure, vocablist)

    print("loading and processing testing data...")
    test = pd.read_csv("data/%s-test.txt" % datatype, sep="\t", header=None)
    output_numeric_reviews(valid, vocablist, "data_processed/%s_test.txt" % datatype)
    test_X, test_y, _ = load_feature_mat(test, procedure, vocablist)

    return train_X, train_y, valid_X, valid_y, test_X, test_y


def f_score(data, y_hat, t):
    if (data == "yelp"):
        f1 = f1_score(t,y_hat, average= "weighted")
    else:
        f1 = f1_score(t,y_hat)
    return f1


# tune Alpha for Binary representation
def ber_bayes(train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    alpha_list = [1e-3, 1e-2, 1e-1, 1]
    f1_scores =[]
    for alpha in alpha_list:
        print("current alpha is: ", alpha)
        bnb = BernoulliNB(alpha=alpha)
        bnb = bnb.fit(train_X, train_y)
        y_hat_valid = bnb.predict(valid_X)
        f1 = f_score(data, y_hat_valid, valid_y)
        print("f1 score: ",f1)
        f1_scores.append(f1)

    best_alpha = alpha_list[f1_scores.index(max(f1_scores))]
    bnb = BernoulliNB(alpha=best_alpha)
    bnb = bnb.fit(train_X, train_y)
    y_hat_test = bnb.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    y_hat_train = bnb.predict(train_X)
    f1_train = f_score(data, y_hat_train, train_y)
    return best_alpha, f1_test, f1_train


# for frequency representation, no need to tune hyperparameter
def gau_bayes(train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    gnb = GaussianNB()
    gnb = gnb.fit(train_X, train_y)
    y_hat_train = gnb.predict(train_X)
    f1_train = f_score(data, y_hat_train, train_y)
    y_hat_valid = gnb.predict(valid_X)
    f1_valid = f_score(data, y_hat_valid, valid_y)
    y_hat_test = gnb.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    return f1_train, f1_valid, f1_test


################################################
### Classfiers: random, majority, decision_trees, and svm
################################################

def random(train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    rd = DummyClassifier(strategy="uniform")
    rd.fit(train_X, train_y)
    y_hat_train = rd.predict(train_X)
    f1_train = f_score(data, y_hat_train, train_y)
    y_hat_valid = rd.predict(valid_X)
    f1_valid = f_score(data, y_hat_valid, valid_y)
    y_hat_test = rd.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    return f1_train, f1_valid, f1_test


def majority(train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(train_X, train_y)
    y_hat_train = majority_clf.predict(train_X)
    f1_train = f_score(data, y_hat_train, train_y)
    y_hat_valid = majority_clf.predict(valid_X)
    f1_valid = f_score(data, y_hat_valid, valid_y)
    y_hat_test = majority_clf.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    return f1_train, f1_valid, f1_test


def decision_trees(train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    criterion_list = ['gini', 'entropy']
    max_depth_list = [3,4,5,6,7,8,9,15,20,25]
    best_f1 = 0
    best_crit = 'gini'
    best_depth = 3
    for criterion in criterion_list:
        for depth in max_depth_list:
            print("criterion is: ", criterion, " max_depth: ", depth)
            dt = DecisionTreeClassifier(criterion=criterion, max_depth= depth)
            dt.fit(train_X, train_y)
            y_hat_valid = dt.predict(valid_X)
            f1 = f_score(data, y_hat_valid, valid_y)
            print("f1 score: ", f1)
            if f1>= best_f1:
                best_f1 = f1
                best_crit = criterion
                best_depth = depth
                best_tr = dt
    y_hat_train = best_tr.predict(train_X)
    f1_train = f_score(data,y_hat_train, train_y)
    y_hat_valid = best_tr.predict(valid_X)
    f1_valid = f_score(data, y_hat_valid, valid_y)
    y_hat_test = best_tr.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    return f1_train, f1_valid, f1_test, best_crit, best_depth


def svm (train_X, train_y, valid_X, valid_y, test_X, test_y, data):
    C_list = [0.01, 0.1, 1, 10, 100]
    best_f1 = 0;
    C_best = 0.01
    for C in C_list:
        print("C is: ", C)
        lin_svm = LinearSVC(C=C, multi_class="ovr")
        lin_svm = lin_svm.fit(train_X, train_y)
        y_hat_valid = lin_svm.predict(valid_X)
        f1 = f_score(data, y_hat_valid, valid_y)
        print("f1 score: ", f1)
        if f1 >= best_f1:
            best_f1 = f1
            best_svm = lin_svm
            C_best = C
    y_hat_train = best_svm.predict(train_X)
    f1_train = f_score(data, y_hat_train, train_y)
    y_hat_valid = best_svm.predict(valid_X)
    f1_valid = f_score(data, y_hat_valid, valid_y)
    y_hat_test = best_svm.predict(test_X)
    f1_test = f_score(data, y_hat_test, test_y)
    return f1_train, f1_valid, f1_test, C_best



def data_procedure(data_name = "yelp", data_represenation = "binary"):
    # load yelp data with bag-of-words representation
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_data(data_name, data_represenation)

    print("=================================================")
    print("random classifier for" + data_name + "represented as" + data_represenation + "bag-of-words")
    f1_train_rd, f1_valid_rd, f1_test_rd = random(train_X, train_y, valid_X, valid_y, test_X, test_y, "yelp")
    print("f1_score on training data", str(f1_train_rd))
    print("f1_score on validation data", str(f1_valid_rd))
    print("f1_score on test data", str(f1_test_rd))
    print("/n")


    print("=================================================")
    print("majority classifier for" + data_name + "represented as" + data_represenation + "bag-of-words")
    f1_train_mj, f1_valid_mj, f1_test_mj = majority(train_X, train_y, valid_X, valid_y, test_X, test_y, "yelp")
    print("f1_score on training data", str(f1_train_mj))
    print("f1_score on validation data", str(f1_valid_mj))
    print("f1_score on test data", str(f1_test_mj))
    print("/n")


    print("=================================================")
    print("naive bayes classifier for" + data_name + "represented as" + data_represenation + "bag-of-words")
    best_alpha, f1_test, f1_train = ber_bayes(train_X, train_y, valid_X, valid_y, test_X, test_y, data="yelp")

    print("best alpha", str(best_alpha), f1_test, f1_train)
    print("f1_score on test data", str(f1_test))
    print("f1_score on training data", str(f1_train))
    print("/n")


    print("=================================================")
    print("decision tree classifier for" + data_name + "represented as" + data_represenation + "bag-of-words")
    f1_train, f1_valid, f1_test, best_crit, best_depth = decision_trees(train_X, train_y, valid_X, valid_y, test_X,
                                                                        test_y, "yelp")
    print("f1_score on training data", str(f1_train))
    print("f1_score on validation data", str(f1_valid))
    print("f1_score on test data", str(f1_test))
    print("best depth", str(best_depth))
    print("/n")


    print("=================================================")
    print("SVM classifier for" + data_name + "represented as" + data_represenation + "bag-of-words")
    f1_train, f1_valid, f1_test, C_best = svm(train_X, train_y, valid_X, valid_y, test_X, test_y, "yelp")

    print("f1_score on training data", str(f1_train))
    print("f1_score on validation data", str(f1_valid))
    print("f1_score on test data", str(f1_test))
    print("best C parameter", str(C_best))
    print("/n")


if __name__ == '__main__':

    print("running data procedure for yelp dataset in binary bag-of-words representation")
    data_procedure("yelp", "binary")

    print("running data procedure for yelp dataset in frequency bag-of-words representation")
    data_procedure("yelp", "frequency")

    print("running data procedure for IMBD dataset in binary bag-of-words representation")
    data_procedure("IMBD", "binary")

    print("running data procedure for IMBD dataset in frequency bag-of-words representation")
    data_procedure("IMBD", "frequency")