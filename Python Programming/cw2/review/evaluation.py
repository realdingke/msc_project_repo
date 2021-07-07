import numpy as np

class Scorer:
    def __init__(self, ground_truth, predictions):
        assert len(ground_truth) == len(predictions), "Length of ground truth and precision" 
        self.ground_truth = np.array(ground_truth)
        self.predictions = np.array(predictions)


    def compute_accuracy(self):
        """ Computes the accuracy score 

        :return: accuracy (float)
        """
       
        # TODO
        #calculate via confusion matrix: TP/toal
        accuracy = np.sum(np.diagonal(self.compute_confusion())) / np.sum(self.compute_confusion())
        return accuracy

 
    def compute_confusion(self):
        """ Computes the confusion matrix

        :return: np.array containing the confusion matrix
        """

        # TODO
        #initialize the confusion matrix as a 2x2 np array
        confusion_matrix = np.zeros((2,2))
        for idx, actual in enumerate(self.ground_truth):
            prediction = self.predictions[idx]
            confusion_matrix[actual][prediction] += 1   #row as actuals, column as predictions
        return confusion_matrix.astype(int)


    def generate_report(self):
        """ Generates a report displaying the confusion matrix and accuracy
        
        The confusion matrix should also show
        - the sum of each row
        - the sum of each column
        - the total number of test instances

        The scorer should return a string that looks like the following:
  
              0    1
         0  814  232 1046
         1  283  765 1048
           1097  997 2094

         Accuracy: 0.7541 

        :return: string
        """
        
        # TODO
        confusion_matrix = self.compute_confusion()
        headings = [0, 1]
        row_sum = np.sum(confusion_matrix, axis=0)
        col_sum = np.sum(confusion_matrix, axis=1)
        col_width = 5 #set the column separation
        output_string = ''
        row_format ="{:>5}" * (len(headings) + 1)
        output_string += row_format.format("", *headings)
        for label, row in zip(headings, confusion_matrix):
            row_string = '\n' + row_format.format(label, *row) + ' ' + str(col_sum[label])
            output_string += row_string
            if label == 1: #reaching last row
                last_row_string = '\n' + row_format.format('', row_sum[0], row_sum[1]) +\
                                  ' ' + str(np.sum(confusion_matrix))
                accuracy_string = '\n' + ' '*(col_width-1) + f'Accuracy: {self.compute_accuracy()}'
                output_string = output_string + last_row_string + '\n' + accuracy_string
        
        return output_string


    def print_report(self):
        """ Prints the report displaying the confusion matrix and accuracy

        Similar to generate_report(), but prints out the string to standard output
        """

        print(self.generate_report())


    def generate_report_to_disk(self, filename):
        """ Generate and save report to a text file on disk 

        Similar to generate_report(), but saves the string to a text file

        :param filename: The filename of the text file to which to save
        """
       
        # TODO
        with open(filename, 'w') as f:
            f.write(self.generate_report())
        pass

 
def _test():
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    #y = np.array([0]*1100 + [1]*200) 
    predictions = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
    #predictions = np.array([0]*1100 + [1]*200)
    #np.random.shuffle(predictions)
    scorer = Scorer(y, predictions)

    print(scorer.compute_accuracy())
    print(scorer.compute_confusion())   
    print(scorer.generate_report())
    #scorer.generate_report_to_disk('test0.txt')


if __name__ == "__main__":
    _test()
