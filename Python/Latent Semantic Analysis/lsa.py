# Latent semantic analysis visualization for a collection of books

from __future__ import print_function, division
from builtins import range

import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

# Download stopwords
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


# stopwords specific to the book title datasets, identified with frequency analysis
books_stopswords = {'introduction', 'edition', 'series', 'application',\
    'approach', 'card', 'access', 'package', 'plus', 'etext',\
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',\
    'second', 'third', 'fourth'}


# combine general tokens and tokens specific to this problem
stopwords_all = stopwords.union(books_stopswords)

book_titles = [line.rstrip() for line in open('./data/all_book_titles.txt')]


def tokenize_words(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    # convert to lower case
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s) # split string into tokens
    tokens = [t for t in tokens if len(t) > 2] # remove short words
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # lemmatize words
    tokens = [t for t in tokens if t not in stopwords_all] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
    return tokens

# create a word-to-index mapping
def load_data(title_collections):
    word_index_map = {}
    current_index = 0
    all_tokens = []
    all_titles = []
    index_word_map = []
    error_count = 0
    for title in title_collections:
        try:
            # throw exception if bad characters
            title = title.encode('ascii', 'ignore').decode('utf-8')
            all_titles.append(title)
            tokens = tokenize_words(title)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in word_index_map:
                    word_index_map[token] = current_index
                    current_index += 1
                    index_word_map.append(token)
        except Exception as e:
            print(e)
            print(title)
            error_count += 1

    print("Number of errors parsing file:", error_count, "number of lines in file:", len(title_collections))
    if error_count == len(title_collections):
        print("No data loaded successfully.  Quitting...")
        exit()
    return all_tokens, word_index_map, index_word_map


# indicator variables in input matrices
def tokens_to_vector(tokens, word_index_map):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x


def get_input_matrix(all_tokens, word_index_map):
    # rows: terms; columns: document
    X = np.zeros((len(word_index_map), len(all_tokens)))
    i = 0
    for tokens in all_tokens:
        X[:,i] = tokens_to_vector(tokens, word_index_map)
        i += 1
    return X


def main():
    all_tokens, word_index_map, index_word_map = load_data(book_titles)
    X = get_input_matrix(all_tokens, word_index_map)
    # load SVD
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(len(word_index_map)):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()

if __name__ == '__main__':
    main()

