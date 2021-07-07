import os
import numpy as np

from review.dataset import MovieReviewDatasetLoader
from review.classification import Classifier, BagOfWordsClassifier, TfidfClassifier
from review.evaluation import Scorer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def set_up_knn():
    """ Returns a KNN classifier 
    :return: an instance of sklearn.neighbors.KNeighborsClassifier
    """

    # TODO
    return KNeighborsClassifier(n_neighbors=5)


def set_up_knn_gridsearchcv():
    """ Sets up GridSearchCV for a KNN classifier
    :return: an instance of sklearn.model_selection.GridSearchCV
    """

    # TODO
    neighbor_nums = [3, 4, 7, 9, 11]  #the K range
    weight_list = ['uniform', 'distance']
    paras_to_tune = dict(n_neighbors=neighbor_nums, weights=weight_list)
    
    gs = GridSearchCV(set_up_knn(), paras_to_tune, cv=10)
    return gs


def test():
    """ Sandbox for you to play around
    """
    pass


def run():
    """ Peforms movie sentiment classification with four classifiers

    This is a demo usage of your classes.
    Please do not edit this function!
    Use test() above instead for your experimental needs!
    """

    dataset_path = "data"
    pos_filename = os.path.join(dataset_path, "rt-polarity.pos")
    neg_filename = os.path.join(dataset_path, "rt-polarity.neg")

    test_size = 0.2
    seed_number = 2020
    max_vocab_size = 10000

    loader = MovieReviewDatasetLoader()
    dataset = loader.load(pos_filename, neg_filename)
    train_dataset, test_dataset = dataset.split_train_test(test_size, seed_number)    

    knn_model = set_up_knn()
    gridsearch_model = set_up_knn_gridsearchcv()
    
    models = [knn_model, gridsearch_model]    
    model_names = ["knn", "gridsearch"]

    report_name_template = "report_{}_{}.txt"

    for (i, model) in enumerate(models):
        print(f"Running Bag of Words + {model}")

        bow_classifier = BagOfWordsClassifier(train_dataset, model, max_vocab_size)
        bow_classifier.train()
        predictions = bow_classifier.predict(test_dataset.X)

        scorer = Scorer(test_dataset.y, predictions)
        scorer.generate_report_to_disk(report_name_template.format("bow", model_names[i]))
        

        print(f"Running TF-IDF + {model}")
        
        tfidf_classifier = TfidfClassifier(train_dataset, model, max_vocab_size)
        tfidf_classifier.train()
        predictions = tfidf_classifier.predict(test_dataset.X)

        scorer = Scorer(test_dataset.y, predictions)
        scorer.generate_report_to_disk(report_name_template.format("tfidf", model_names[i]))


if __name__ == "__main__":
    run()
    # test()

