class Node:
    """
    Abstract Node class, to template for Leaf and Tree nodes
    """
    def __str__(self):
        raise NotImplementedError


class LeafNode(Node):
    """
    Leaf Node class, contains a label
    """
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return "(Leaf " + str(self.label) + ")"

    def decide(self, sample):
        return self.label


class TreeNode(Node):
    """
    Tree Node class, contains data for splitting a given data set and left and
    right subtrees
    """
    def __init__(self, attr, value, lhs, rhs):
        self.attr = attr
        self.value = value
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return (
            "(Node X"
            + str(self.attr)
            + " <= "
            + str(self.value)
            + " "
            + str(self.lhs)
            + " "
            + str(self.rhs)
            + ")"
        )

    def decide(self, sample):
        if sample[self.attr] <= self.value:
            return self.lhs.decide(sample)
        else:
            return self.rhs.decide(sample)
