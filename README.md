# sample_code

Project List

--Python
  --NLP Text Classifier Project
    In this project, I designed a sentiment classifier for classifying the sentiment of the
    reviews. This is a Natural Language Processing (NLP) task where the input is a natural
    language text and output is the sentiment label. 
    I will consider two different review datasets: yelp reviews for restaurants and IMDB reviews for movies.

    yelp dataset--
    The Yelp dataset consists of 7000 reviews in the training set, 1000 reviews in the validation
    set, and 2000 reviews in the test set. This is a 5 class problem where each review is classified
    into one of the five ratings with rating-5 being the best score and rating-1 being the worst
    score.

    IMDB dataset--
    IMDB dataset consists of 15000 reviews in training set, 10000 reviews in validation set, and
    25000 reviews in test set. This is a 2 class problem with class 1 being positive sentiment and
    class 0 being negative sentiment.

    I use the F1-measure as the evaluation metric for the entire assignment. 
    As a baseline, I report the performance of the random classifier (a classifier which classifies
    a review into an uniformly random class) and the majority-class classifier (a classifier
    which computes the majority class in the training set and classifies all test instances as
    that majority class). Then, I train Naive Bayes, Decision Trees, and Linear SVM for this
    task. To tune the model, I did a thorough hyper-parameter tuning by using the given validation
    set.

    In the report, I report the list of hyper-parameters considered for
    each classifier, the range of the individual hyper-parameters and the best value for this
    hyper-parameters chosen based on the validation set performance. I report training,
    validation, and test F1-measure for all the classifiers (with best hyper-parameter configuration), and comment about the performance of different classifiers.



