# Introduction to ML coursework: Decision Tree User Manual

### Entry Point
The entry point of the program is located in the decision_tree.py file located
in the root directory.

Help with running the program with flags can be accessed by running:

    python3 decision_tree.py --help

### Running Cross-Validation Experiments
Experiments can be ran in different configuration using the flags provided to the program. The required parameter is the number of folds to use in cross-validation experiments, this can be given at the end of the command.

To run cross-validation experiments, use the flags `-c` or `-n` to perform the experiment on clean and noisy data sets respectively.

For example

    python3 decision_tree.py -c 10

will run 10-fold cross-validation on the clean data set. At the end of each experiment, the confusion matrix, class averaged precision, recall, F1-measure and average classification rate are printed out.

### Pruning
Pruning can be activated by passing the `-p` or `--prune` flag to the program along with a target data set as per the previous section to run the standard cross-validation experiment as well as cross-validation on the same data set with pruning.

    python3 decision_tree.py -c -p 10

This will run both experiments, printing out the results. Comparisons can be made between the performance of the two approaches this way.

### Running All Experiments
The program can be ran with the `--all` flag to run cross-validation experiments without and with pruning in sequence on both the clean and noisy data sets. This action forgoes the need for clean, dirty and pruning flags though the fold number is still required.

### Visualising Decision Trees
In addition to the aforementioned experiments on the performance of decision tree algorithms, decision trees can be visualised by
passing the `-sc` and `-sn` flags which will run separate individual experiments, training and plotting a single decision tree. This can be combined with the `-p` flag for visual comparisons of the effects of the pruning process.

For example:

    python3 decision_tree.py -sc -p 10

will train a single decision tree on the clean data set with 9:1 training-validation split, plotting the tree before and after pruning.

If the `-p` flag is not given, the single tree is trained on the full data set.

This is not included under `--all` flag as it requires user interaction with the generated plot to proceed and may cause issues when running on lab computers over ssh.

### Adding Your Own Data Files
To add your own data files, specify the path in `src/paths.py`, existing
examples are used in `decision_tree.py` for the regular experiments.

### Example

    python3 decision_tree.py -c -n -p -sc -sn 10

This will run 10-fold cross-validation experiment on both clean and noisy data sets, then run again with pruning, printing out the performance of each experiment. After which a single decision tree is trained on a randomised clean data set, and plotted before pruning and repeated plotting. The process is then repeated for the noisy data set.