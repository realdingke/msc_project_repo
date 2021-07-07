import os
from src.decision_node import Node, TreeNode, LeafNode

if "DISPLAY" not in os.environ:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_HEIGHT = 100
LINE_WIDTH = 100


def depth(tree: Node):
    """
    Get the depth of the input node.
    """

    if isinstance(tree, LeafNode):
        return 1
    else:
        return max(depth(tree.lhs), depth(tree.rhs)) + 1


def plot_node(x, y, node: Node):
    """
    Plot a node on at coordinates (x,y).
    """

    if isinstance(node, LeafNode):
        text = f"leaf: {node.label}"
    else:
        text = f"{node.attr} <= {node.value}"

    plt.text(
        x,
        y,
        text,
        bbox={"facecolor": "white"},
        horizontalalignment="center",
        verticalalignment="center",
    )


def tree_plot_helper(x, y, width, tree: Node):
    """
    Plots a given node at (x,y) and it's left and right edges if it has
    children. Then recursively calls itself on child nodes.
    """

    # Plot the current node
    plot_node(x, y, tree)

    # Plot edges and children.
    if isinstance(tree, TreeNode):
        end_y = y - LINE_HEIGHT

        if tree.lhs:
            end_x = x - width

            plt.plot([x, end_x], [y, end_y], "k")
            tree_plot_helper(end_x, end_y, width / 2, tree.lhs)

        if tree.rhs:
            end_x = x + width

            plt.plot([x, end_x], [y, end_y], "k")
            tree_plot_helper(end_x, end_y, width / 2, tree.rhs)


def tree_plot(tree: Node):
    """
    Visualize a tree.
    """

    tree_depth = depth(tree)

    height = tree_depth * LINE_HEIGHT
    width = (2 ** (tree_depth - 1)) * LINE_WIDTH

    # Root node coordinates: (start_x, start_y)
    start_x = width / 2
    start_y = height

    # Plot root node and children
    tree_plot_helper(start_x, start_y, width / 2, tree)

    plt.axis("off")

    if "DISPLAY" in os.environ:
        plt.show()
    else:
        print("Please enter filename for the output plot file (without extension):")
        filename = input()
        plt.savefig(f"{filename}.png")
