from src.decision_node import LeafNode, TreeNode
from src.confusion_matrix import ConfusionMatrix
import numpy as np
import random

from src.tree_plot import tree_plot

NUM_CLASSES = 4


# FUNCTIONS GENERATING DECISION TREE ===========================================


def _same_labels(dataset):
    """
    Determines whether the given data set entries all have the same label
    """
    if len(dataset) > 0:
        first_label = dataset[0][7]
        for entry in dataset:
            if entry[7] != first_label:
                return False
    return True


def _entropy(dataset):
    """
    Calculate entropy of a given dataset
    """
    last_column = dataset[:, -1]
    _, label_counts = np.unique(last_column, return_counts=True)

    p = label_counts / label_counts.sum()

    return sum(p * (-np.log2(p)))


def _gain(dataset=None, left=None, right=None, entropy_all=-1):
    """
    Calculate the information gain between the current data set and its split
    parts
    """
    if left is not None and right is not None:
        len_a = len(dataset)
        len_l = len(left)
        len_r = len(right)
        if entropy_all < 0:
            entropy_all = _entropy(dataset)
        return entropy_all - (
            ((len_l / len_a) * _entropy(left)) + ((len_r / len_a) * _entropy(right))
        )
    return None


def _find_split(dataset):
    """
    Finds a value to split the data set by
    """
    max_ig = [0, -1, -1, None, None]
    e_all = _entropy(dataset)
    for i in range(len(dataset[0]) - 1):
        sorted_copy = dataset[np.ndarray.argsort(dataset[:, i])]
        for j in range(1, len(sorted_copy)):
            if sorted_copy[j][i] != sorted_copy[j - 1][i]:
                ig = _gain(
                    dataset,
                    entropy_all=e_all,
                    left=sorted_copy[:j],
                    right=sorted_copy[j:],
                )
                if ig > max_ig[0]:
                    max_ig = [
                        ig,
                        i,
                        ((sorted_copy[j][i] + sorted_copy[j - 1][i]) / 2),
                        sorted_copy[:j],
                        sorted_copy[j:],
                    ]

    [_, attr, value, left, right] = max_ig
    # return split as an attr, value pair, where i <= v is the condition
    return attr, value, left, right


def decision_tree_learning(dataset, depth):
    """
    Trains a single decision tree
    :returns decision tree, depth of tree
    """
    if _same_labels(dataset):
        # Return a root node
        return LeafNode(label=int(dataset[0][7])), depth
    else:
        (attr, a_value, l_dataset, r_dataset) = _find_split(dataset)
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
        node = TreeNode(attr, a_value, l_branch, r_branch)
        return node, max(l_depth, r_depth)


# FUNCTION CROSS VALIDATION ====================================================


def print_confusion_matrix_stats(confusion_matrix):
    """
    Prints the confusion matrix, precisions, recalls, F1-measure and
    micro-average classification rate from the given confucsion matrix
    """
    print("Confusion matrix is:")
    print(confusion_matrix)
    print("Precisions are:")
    precisions = confusion_matrix.get_precisions()
    print(precisions)
    print("Recalls are:")
    recalls = confusion_matrix.get_recalls()
    print(recalls)
    print("F1 scores are:")
    print(np.divide(np.multiply(2 * precisions, recalls), precisions + recalls))
    print("Average classification rate is:")
    print(confusion_matrix.get_classification_rate())


def generate_confusion_matrix(test_db, trained_tree):
    """
    Generate a confusion matrix for a given
    """
    confusion_matrix = np.zeros((4, 4))

    for sample in test_db:
        result = trained_tree.decide(sample)
        confusion_matrix[result - 1][int(sample[7]) - 1] += 1

    return ConfusionMatrix(confusion_matrix)


def cross_validate(data, fold_num):
    """
    Perform cross validation with fold_num number of folds
    """
    np.random.shuffle(data)
    folds = np.array_split(data, fold_num)

    # Initialize the confusion matrix with zeros for summing
    confusion_matrix_sum = ConfusionMatrix.zero_confusion_matrix(NUM_CLASSES)
    for test_id in range(fold_num):
        print(
            f"Training and evaluating decision tree {test_id + 1} / {fold_num}",
            end="\r",
            flush=True,
        )
        train_data = np.concatenate(np.delete(folds, test_id, axis=0))
        test_data = folds[test_id]

        decision_tree, _ = decision_tree_learning(train_data, 0)
        confusion_matrix_sum = confusion_matrix_sum + generate_confusion_matrix(
            test_data, decision_tree
        )
    print()

    # Average confusion matrix
    confusion_matrix = confusion_matrix_sum / fold_num

    print_confusion_matrix_stats(confusion_matrix)


# FUNCTION PRUNING =============================================================


def _splitArray(arr, cond):
    return arr[cond], arr[~cond]


# Split a dataset to feature values xi less than and greater than the node's
def split_at_node(node, dataset):
    return _splitArray(dataset, dataset[:, node.attr] <= node.value)


def prune_singular(current_node, validation_subset):
    """
    Prune a single decision tree using the given validation data set
    """
    labels = validation_subset[:, 7]
    if isinstance(current_node, LeafNode):
        # Already is leaf, do not change
        return current_node, (labels == current_node.label).sum()
    elif isinstance(current_node, TreeNode):
        # Prune lhs and rhs before checking if current node should be pruned
        lhs_data, rhs_data = split_at_node(current_node, validation_subset)
        lhs, lhs_score = prune_singular(current_node.lhs, lhs_data)
        rhs, rhs_score = prune_singular(current_node.rhs, rhs_data)

        if isinstance(lhs, LeafNode) and isinstance(rhs, LeafNode):
            # Node is eligible for pruning
            if validation_subset.shape[0] == 0:
                # In event where no data point from validation data set reaches
                # this node, we prune and select random label from its children
                rand_label = [lhs.label, rhs.label][bool(random.getrandbits(1))]
                return LeafNode(rand_label), 0

            # If there is valid data to select from, we select the majority
            # label within the validation data subset as our new leaf label
            label_counts = [(labels == i).sum() for i in range(1, 5)]
            majority = [lhs.label, rhs.label][
                np.argmax([label_counts[lhs.label - 1], label_counts[rhs.label - 1]])
            ] - 1

            # If the new majority label provides a better percentage of correct
            # predictions than the existing children trees, we prune
            if label_counts[majority] >= (lhs_score + rhs_score):
                return LeafNode(majority + 1), label_counts[majority]

        # Current node is not eligible for pruning
        return TreeNode(current_node.attr, current_node.value, lhs, rhs), (lhs_score + rhs_score)
    else:
        raise ValueError


def cross_validate_and_prune(data, fold_num):
    """
    Perform cross validation within cross validation to train and prune
    """
    np.random.shuffle(data)
    folds = np.array_split(data, fold_num)

    # Initialize the confusion matrix with zeros for summing
    confusion_matrix_sum = ConfusionMatrix.zero_confusion_matrix(NUM_CLASSES)
    for test_id in range(fold_num):

        test_data = folds[test_id]
        remaining_data = np.delete(folds, test_id, axis=0)
        for prune_id in range(fold_num - 1):
            print(
                f"Pruning tree {prune_id + 1}/{fold_num - 1}"
                + f" in fold {test_id + 1}/{fold_num}",
                end="\r",
                flush=True,
            )
            validation_data = remaining_data[prune_id]
            training_data = np.concatenate(np.delete(remaining_data, prune_id, axis=0))
            decision_tree, max_depth = decision_tree_learning(training_data, 0)
            pruned_tree, _ = prune_singular(decision_tree, validation_data)
            confusion_matrix_sum = confusion_matrix_sum + generate_confusion_matrix(
                test_data, pruned_tree
            )
    print()

    # Average confusion matrix
    confusion_matrix = confusion_matrix_sum / (fold_num * (fold_num - 1))

    print_confusion_matrix_stats(confusion_matrix)


# FUNCTION GENERATE AND PLOT TREE ==============================================


def train_and_plot_tree(dataset, fold_num, prune=False):
    """
    Trains a single decision tree using the given data set and optionally prunes
    it, plots the result(s)
    """
    np.random.shuffle(dataset)

    training_data = dataset
    if prune:
        folds = np.array_split(dataset, fold_num)
        validation_data = folds[0]
        training_data = np.concatenate(folds[1:])

    decision_tree, _ = decision_tree_learning(training_data, 0)
    tree_plot(decision_tree)

    if prune:
        pruned_tree, _ = prune_singular(decision_tree, validation_data)
        tree_plot(pruned_tree)
