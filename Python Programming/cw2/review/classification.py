from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Classifier:
    """ A base Classifier class
    """

    def __init__(self, training_dataset, model=None):
        """ Constructor for Classifier.

        :param training_dataset: a Dataset instance for training 
        :param model: an estimator that implements fit() and predict() methods
        """

        self._dataset = training_dataset
        self._model = model


    def train(self):
        """ Trains the classifier
        """

        self._model.fit(self._dataset.X, self._dataset.y)
    

    def predict(self, X):
        """ Performs predictions given the samples

        :param X: feature vectors X to predict
        :return: a numpy array containing the predicted labels.
        """

        return self._model.predict(X)



class BagOfWordsClassifier(Classifier):
    def __init__(self, training_dataset, base_model, max_vocab_size=10000):
        """ Constructor for BagOfWordsClassifier.

        :param training_dataset: a Dataset instance for training
        :param base_model: an estimator that implements fit() and predict() methods
        :param max_vocab_size: maximum number of words allowed in the vocabulary
        """
        
        super().__init__(training_dataset)
        self.max_vocab_size = max_vocab_size
        self._set_up_pipeline(base_model)


    def _set_up_pipeline(self, base_model):
        """ Sets up the bag of words classifier
         
        Sets up the model as a pipeline that first transforms the raw instances into a bag of words representation,
        followed by the given base estimator.
        """
 
        # TODO
        self._model = Pipeline(steps=[('transformer', CountVectorizer(max_features=self.max_vocab_size)),
                                      ('estimator', base_model)])
        
        pass


class TfidfClassifier(BagOfWordsClassifier):
    def __init__(self, training_dataset, base_model, max_vocab_size=10000):
        """ Constructor for TfidfClassifier.

        :param training_dataset: a Dataset instance for training
        :param base_model: an estimator that implements fit() and predict() methods
        :param max_vocab_size: maximum number of words allowed in the vocabulary
        """

        super().__init__(training_dataset, base_model, max_vocab_size)
        self._set_up_pipeline(base_model)


    def _set_up_pipeline(self, base_model):
        """ Sets up the Tf-idf classifier
         
        Sets up the model as a pipeline that first transforms the raw instances into a tf-idf weighted feature vector,
        followed by the given base estimator.
        """

        # TODO
        self._model = Pipeline(steps=[('transformer', TfidfVectorizer(max_features=self.max_vocab_size)),
                                      ('estimator', base_model)])
        pass



def _test_bow_classifier():
    from dataset import MovieReviewDatasetLoader  
    loader = MovieReviewDatasetLoader()
    dataset = loader.load("data/rt-polarity.pos", "data/rt-polarity.neg")
    
    train_dataset, test_dataset = dataset.split_train_test(test_size=0.2)    

    from sklearn.svm import SVC
    model = SVC()
    classifier = BagOfWordsClassifier(train_dataset, model, max_vocab_size=5000) 
    classifier.train()

    predictions = classifier.predict(test_dataset.X)
    print(predictions)    


if __name__ == "__main__":
    _test_bow_classifier()
