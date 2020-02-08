from utlis import class_counts,find_best_split,partition,Question
import random

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


#######
# Demo:
# The tree predicts the 1st row of our
# training data is an apple with confidence 1.
# my_tree = build_tree(training_data)
# classify(training_data[0], my_tree)
#######

def predict(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    for lbl in counts.keys():
        probs= counts[lbl] / total
        if probs >= 0.49999:
            return lbl

def score(test,model):
    result = []
    for row in test:
        result.append((row[-1], predict(classify(row, model))))
    precision = sum([1 for x in result if x[0] == x[1]]) / len(test)
    return precision

def dispatch_index(index_list, data_list):
    return [data_list[index] for index in index_list]


def train_test_dispatch(rows,train_index, test_index):
    train_data = dispatch_index(train_index,rows)
    test_data = dispatch_index(test_index,rows)
    return train_data,test_data

def five_cv_score(rows,model):
    length = len(rows)
    length_test = int(length * 0.2)
    score_list = []
    for i in range(5):
        train_index = []
        test_index = []
        while len(test_index) <= length_test:
            test_index.append(random.randrange(0, length - 1))
            test_index = list(set(test_index))
        train_index = list(set(range(length)) - set(test_index))
        train_data, test_data = train_test_dispatch(rows,train_index,test_index)
        dt = model(train_data)
        score_list.append(score(test_data,dt))
    print(sum(score_list)/len(score_list))
    return sum(score_list)/len(score_list)