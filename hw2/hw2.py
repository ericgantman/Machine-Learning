import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
             0.25: 1.32,
             0.1: 2.71,
             0.05: 3.84,
             0.0001: 100000},
             2: {0.5: 1.39,
             0.25: 2.77,
             0.1: 4.60,
             0.05: 5.99,
             0.0001: 100000},
             3: {0.5: 2.37,
             0.25: 4.11,
             0.1: 6.25,
             0.05: 7.82,
             0.0001: 100000},
             4: {0.5: 3.36,
             0.25: 5.38,
             0.1: 7.78,
             0.05: 9.49,
             0.0001: 100000},
             5: {0.5: 4.35,
             0.25: 6.63,
             0.1: 9.24,
             0.05: 11.07,
             0.0001: 100000},
             6: {0.5: 5.35,
             0.25: 7.84,
             0.1: 10.64,
             0.05: 12.59,
             0.0001: 100000},
             7: {0.5: 6.35,
             0.25: 9.04,
             0.1: 12.01,
             0.05: 14.07,
             0.0001: 100000},
             8: {0.5: 7.34,
             0.25: 10.22,
             0.1: 13.36,
             0.05: 15.51,
             0.0001: 100000},
             9: {0.5: 8.34,
             0.25: 11.39,
             0.1: 14.68,
             0.05: 16.92,
             0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    # Last column, which containes the lables
    labels = data[:, -1]
    # Number of instances
    total_instances = len(labels)
    # Get the unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # Calculate the probability of each label
    label_probabilities = label_counts / total_instances
    # Calculate the Gini impurity - Sigma
    for probability in label_probabilities:
        gini += probability ** 2
    gini = 1 - gini
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    # Last column, which containes the lables
    labels = data[:, -1]
    # Number of instances
    total_instances = len(labels)
    # Get the unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # Calculate the probability of each label
    label_probabilities = label_counts / total_instances
    # Calculate the entropy
    for probability in label_probabilities:
        if probability != 0:  # Avoid log(0)
            entropy -= probability * np.log2(probability)
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        # Last column, which containes the lables
        labels = self.data[:, -1]
        # Get the unique labels and their counts
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        # prediction is the most common class label
        pred = unique_labels[np.argmax(label_counts)]
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)  # add child node to list of children
        self.children_values.append(val)  # add value associated with the child

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        parent_impurity = (len(self.data) / n_total_sample) * \
            self.impurity_func(self.data)
        child_impurity = 0
        for child in self.children:
            child_impurity += (len(child.data) / n_total_sample) * \
                self.impurity_func(child.data)
        self.feature_importance = parent_impurity - child_impurity

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        if self.gain_ratio:
            self.impurity_func = calc_entropy
        impurity_data = self.impurity_func(self.data)
        # Node already pure
        if impurity_data == 0:
            return goodness, groups
        # the values for this feature
        values, values_count = np.unique(
            self.data.T[feature], return_counts=True)

        # In case all feature values are the same for all instances there will be no improve in impurity
        # and the goodness of split will be 0
        if len(values) == 1:
            return goodness, groups

        p = values_count / len(self.data)
        # split the data into groups according values
        groups = {value: self.data[self.data[:, feature] == value]
                  for value in values}
        weighted_average_impurity_child = 0
        for i in range(len(p)):
            weighted_average_impurity_child += p[i] * \
                self.impurity_func(groups[values[i]])
        goodness = impurity_data - weighted_average_impurity_child

        if self.gain_ratio:  # If it will set to True it will return the Gain Ratio.
            split_information = -p.dot(np.log2(p))
            # divide by zero can occur when there is only one value and we cover this case up
            goodness /= split_information

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.depth >= self.max_depth:
            # Reached maximum depth, stop splitting
            self.terminal = True
            return

        # Calculate the goodness of split or gain ratio for each feature
        best_feature = -1
        best_goodness = -1
        best_groups = {}

        # Exclude the last column (labels)
        for feature_idx in range(len(self.data[0]) - 1):
            goodness, groups = self.goodness_of_split(feature_idx)
            # Update the best feature and goodness if the current feature is better
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature_idx
                best_groups = groups

        # In case the best_goodness is 0 no need for split and make the node leaf.
        if best_goodness == 0:
            self.terminal = True
            return
        # Check if splitting should be performed based on pruning conditions
        if self.chi < 1:
            if not self._chi_square_test(best_groups):
                self.terminal = True
                return

        self.feature = best_feature
        for value in best_groups.keys():
            child = DecisionNode(best_groups[value], self.impurity_func, depth=self.depth + 1,
                                 chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)

    def _chi_square_test(self, best_groups):
        labels, count_labels = np.unique(self.data[:, -1], return_counts=True)
        n_labels = len(labels)
        n_values = len(best_groups)
        p_y = count_labels / len(self.data)

        d = np.array([len(best_groups[key]) for key in best_groups])
        observed_counts = np.zeros((n_values, n_labels))
        expected_counts = np.zeros((n_values, n_labels))

        # Fill the observed counts array
        for i, (key, group) in enumerate(best_groups.items()):
            for j, label in enumerate(labels):
                observed_counts[i, j] = np.count_nonzero(group[:, -1] == label)

        # Compute expected counts for each label in each group
        for j in range(n_labels):
            expected_counts[:, j] = d * p_y[j]

        # Compute the chi-square statistic
        # Handle divisions by zero if expected counts are zero
        with np.errstate(divide='ignore', invalid='ignore'):
            chi_squares = (observed_counts -
                           expected_counts) ** 2 / expected_counts
            # Replace NaN results with zero
            chi_squares[np.isnan(chi_squares)] = 0

        chi_square_stat = chi_squares.sum()
        df = (n_values - 1) * (n_labels - 1)

        # Compare the chi-square statistic to the critical value from the chi-square distribution
        critical_value = chi_table[df][self.chi]

        return chi_square_stat > critical_value

    def tree_depth(self):
        max_depth_of_childs = 0
        if self.terminal:
            return 0
        else:
            for child in self.children:
                child_depth = child.tree_depth()
                if child_depth > max_depth_of_childs:
                    max_depth_of_childs = child_depth
            return max_depth_of_childs + 1


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        self.build_tree_recursive(self.root)

    def build_tree_recursive(self, node):
        node.split()
        node.calc_feature_importance(len(self.data))
        for child in node.children:
            self.build_tree_recursive(child)

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        node = self.root
        while not node.terminal:
            value = instance[node.feature]
            if value not in node.children_values:
                return node.pred
            index = node.children_values.index(value)
            node = node.children[index]
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct_predictions = 0
        for instance in dataset:
            pred = self.predict(instance)
            if pred == instance[-1]:
                correct_predictions += 1
        accuracy = (correct_predictions / len(dataset))
        return accuracy

    def depth(self):
        return self.root.tree_depth()


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    tree = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Initialize and train the decision tree
        tree = DecisionTree(
            X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        # Append accuracies to lists
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    pruning_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi_value in pruning_values:
        # Build decision tree with current chi value
        tree = DecisionTree(X_train, impurity_func=calc_entropy,
                            chi=chi_value, gain_ratio=True)
        tree.build_tree()

        # Record tree depth
        depth.append(tree.depth())
        # Calculate training accuracy
        training_accuracy = tree.calc_accuracy(X_train)
        chi_training_acc.append(training_accuracy)

        # Calculate validation accuracy
        validation_accuracy = tree.calc_accuracy(X_test)
        chi_validation_acc.append(validation_accuracy)

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """

    if node is None:
        return 0
    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    return n_nodes
