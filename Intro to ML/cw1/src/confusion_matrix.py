import numpy as np


class ConfusionMatrix:
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix

    @staticmethod
    def zero_confusion_matrix(n):
        """Returns a zero matrix with dimensions n x n"""
        return ConfusionMatrix(np.zeros((n, n)))

    def __str__(self):
        return str(self.confusion_matrix)

    def __add__(self, other):
        return ConfusionMatrix(self.confusion_matrix + other.confusion_matrix)

    def __truediv__(self, constant):
        return ConfusionMatrix(self.confusion_matrix / constant)

    def get_precisions(self):
        """Returns a vector of precisions for each class"""
        true_positives = np.diagonal(self.confusion_matrix)
        classifications = np.sum(self.confusion_matrix, axis=1)

        return true_positives / classifications

    def get_recalls(self):
        """Returns a vector of recalls for each class"""
        true_positives = np.diagonal(self.confusion_matrix)
        totals = np.sum(self.confusion_matrix, axis=0)

        return true_positives / totals

    def get_classification_rate(self):
        """Returns an overall classification rate"""
        true_positives = np.sum(np.diagonal(self.confusion_matrix))
        total = np.sum(self.confusion_matrix)

        return true_positives / total
