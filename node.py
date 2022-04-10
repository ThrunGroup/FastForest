from tree import Tree


class Node:
    def __init__(self, tree: Tree, depth: int):
        self.tree = tree  # The tree this node is inside
        self.data = self.tree.data  # TODO(@motiwari): Is this a reference or a copy?
        self.depth = depth
        self.left = None
        self.right = None
        self.zeros = 0
        self.ones = 0
        self.last_feature_split = None
        self.next_feature_split = None
        self.best_split = None

    def calculate_best_split(self):
        """
        Speculatively calculate the best split
        :return: None, but assign
        """
        pass

    def split(self):
        """
        Splits the node into two children nodes.
        :return: None
        """
        if self.best_split is None:
            self.calculate_best_split()

        ## Perform split



