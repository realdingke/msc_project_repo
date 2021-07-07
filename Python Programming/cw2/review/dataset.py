import numpy as np

class Dataset:
    def __init__(self, X=np.zeros(0), y=np.zeros(0)):
        """ Constructor for class Dataset.

        :param X: an iterable containing the samples
        :param y: an iterable containing the labels        
        """
        
        assert len(X) == len(y), f"X and y must be of equal length. Got len(X): {len(X)}, len(y): {len(y)}"
        self.__X = np.array(X)
        self.__y = np.array(y) 


    @property 
    def X(self):
        return self.__X


    @property
    def y(self):
        return self.__y


    def __len__(self):
        return len(self.X)
        

    def __add__(self, other):
        """ Allows two Dataset instances be concatenated. 
        
        Returns a new Dataset instance by concatenating self and other
          
        :param other: a Dataset instance to concatenate to this one
        :return: a new Dataset instance
        """

        # TODO
        add_X = np.concatenate((self.X, other.X))
        add_y = np.concatenate((self.y, other.y))
        
        return Dataset(X=add_X, y=add_y)
       
 
    def append(self, other):
        """ Appends the entries of a new Dataset onto this Dataset instance. 

        This is a mutator method, so should modify the existing Dataset instance.
        It does not return anything.

        :param other: a Dataset instance to concatenate to this one
        """

        # TODO
        # Hint: This is similar to __add__(), except that instead of returning the values
        # you update the values directly
        # Remember that self.X and self.Y are read-only, so you cannot set these directly
        self.__X = np.concatenate((self.X, other.X))
        self.__y = np.concatenate((self.y, other.y))
        pass 


    def split_train_test(self, test_size=0.2, seed_number=42):
        """ Splits the data into training/testing data and labels

        :param test-size: the percentage of the data that should be used for testing
        :return: a tuple with 2 elements (dataset_train, dataset_test).
                 - dataset_train is a Dataset instance containing the training examples
                 - dataset_test is a Dataset instance containing the test examples
        """
        
        # TODO
        # You may use sklearn for this
        np.random.seed(seed_number)
        np.random.shuffle(self.__X)
        np.random.seed(seed_number)
        np.random.shuffle(self.__y)
        #split X
        test_size_X = int(np.round_(test_size * len(self.__X), 1))
        test_X = self.__X[:test_size_X]
        train_X = self.__X[test_size_X:]
        #split y
        test_size_y = int(np.round_(test_size * len(self.__y), 1))
        test_y = self.__y[:test_size_y]
        train_y = self.__y[test_size_y:]
        return (Dataset(X=train_X, y=train_y), Dataset(X=test_X, y=test_y))


class DatasetLoader:
    """ A DatasetLoader for loading a dataset instance 
    """

    def load(self, filename, label=0):
        """ Loads a Dataset instance from a filename.

        Will assign all N rows in the file to a specified label.

        :param filename: path to the file containing the samples, one per line
        :param label: a label to be assigned to all samples
        :return: an instance of a Dataset class
        """

        # TODO
        with open(filename) as f:
            lines = f.readlines()
        #populate the feature data, labels as lists
        feature_list = []
        label_list = []
        for line in lines:
            feature_list.append(line.strip())
            label_list.append(label)
        
        return Dataset(X=feature_list, y=label_list)

class MovieReviewDatasetLoader(DatasetLoader):
    """ A DatasetLoader specifically for loading the movie review dataset
    """
    
    def load(self, pos_filename, neg_filename):
        """ Loads the dataset from one file containing positive examples and another containing negative examples.
        
        Returns a Dataset instance with the contatenation of instances from both files, and with the label 0 asigned to negative examples and label 1 assigned to positive examples.

        :param pos_filename: path to the file containing positive examples, one per line
        :param neg_filename: path to the file containing negative examples, one per line

        :return: a Dataset instance
        """
       
        # TODO
        # Hint: Don't Repeat Yourself. Reuse the load() method from the superclass DatasetLoader by calling super().load()
        pos_data = super().load(filename=pos_filename, label=1)
        neg_data = super().load(filename=neg_filename, label=0)
        combined_data = pos_data + neg_data  #use + overload
        return combined_data


def _test_dataset():
    X = ["abc", "bcd", "cde", "def"]
    y = [0, 0, 1, 1]
    dataset1 = Dataset(X, y) 
    print(dataset1.X)
    print(dataset1.y)

    X2 = ["123", "456", "789"]
    y2 = [0, 1, 0]
    dataset2 = Dataset(X2, y2)
    print(dataset2.X)
    print(dataset2.y)

    dataset3 = dataset1 + dataset2
    print(dataset3.X)  # ["abc", "bcd", "cde", "def", "123", "456", "789"] 
    print(dataset3.y)  # [0, 0, 1, 1, 0, 1, 0]

    dataset1.append(dataset2)
    print(dataset1.X)  # ["abc", "bcd", "cde", "def", "123", "456", "789"] 
    print(dataset1.y)  # [0, 0, 1, 1, 0, 1, 0]

    X = ["abc"]*20 + ["def"]*20
    y = [0]*20 + [1]*20
    dataset = Dataset(X, y) 
    print(dataset.X)
    print(dataset.y)
 
    dataset_train, dataset_test = dataset.split_train_test()
    print(dataset_train.X)
    print(dataset_train.y)
    print(dataset_test.X)
    print(dataset_test.y)


def _test_dataset_loader():
    loader = DatasetLoader()
    dataset = loader.load("data/rt-polarity.pos", 1)
    print(dataset.X)
    print(dataset.y)


def _test_review_loader():
    loader = MovieReviewDatasetLoader()
    dataset = loader.load("data/rt-polarity.pos", "data/rt-polarity.neg")
    print(dataset.X)
    print(dataset.y)

if __name__ == "__main__":
    _test_dataset()
    _test_dataset_loader()
    _test_review_loader()
