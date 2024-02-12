import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def histograms(matr):
    right=[]
    wrong=[]
    for i in range(len(matr[:])):
        S=0
        for j in range(len(matr[0])):
            if i==j:
                right.append(matr[i][j])
            else:
                S+=matr[i][j]
        wrong.append(S)
    return right, wrong



class Node():
    def __init__(self, attribute_of_vector=None, max_value=None, left=None, right=None, information_gain=None, class_name=None):
        
        self.attribute_of_vector = attribute_of_vector
        self.max_value = max_value
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.class_name = class_name


class DecisionTree():
    def __init__(self, criterion, min_vectors=3, max_depth=25):
        self.root = None
        self.min_vectors = min_vectors
        self.max_depth = max_depth
        self.criterion=criterion
    def build_tree(self, X, Y, current_depth=0):


        num_of_vectors, num_of_attributes = np.shape(X)
        if num_of_vectors >= self.min_vectors and current_depth <= self.max_depth:
            
            best_split = self.get_best_split(X, Y, num_of_attributes)
            
            if best_split["information_gain"] > 0:
                
                left_subtree = self.build_tree(best_split["x_left"], best_split["y_left"], current_depth + 1)
                
                right_subtree = self.build_tree(best_split["x_right"], best_split["y_right"], current_depth + 1)
               
                return Node(best_split["attribute_of_vector"], best_split["max_value"],
                            left_subtree, right_subtree, best_split["information_gain"])


        leaf_class_name = max(list(Y), key=list(Y).count)

        return Node(class_name=leaf_class_name)

    def get_best_split(self, X, y, num_of_attributes):
        
        best_split = {}
        max_information_gain = -1
      
        for attribute_of_vector in range(num_of_attributes):
            attribute_values = X[:, attribute_of_vector]
            possible_max_values = np.unique(attribute_values)
        
            for max_value in possible_max_values:

                x_left, x_right, left_y, right_y = self.split(X, y, attribute_of_vector, max_value)
                
                if len(x_left) > 0 and len(x_right) > 0:

                    current_information_gain = self.information_gain(y, left_y, right_y)
                    
                    if current_information_gain > max_information_gain:
                        best_split["attribute_of_vector"] = attribute_of_vector
                        best_split["max_value"] = max_value
                        best_split["x_left"] = x_left
                        best_split["x_right"] = x_right
                        best_split["y_left"] = left_y
                        best_split["y_right"] = right_y
                        best_split["information_gain"] = current_information_gain
                        max_information_gain = current_information_gain

        return best_split

    def split(self, x, y, attribute_of_vector, max_value):
        x_left=[]
        x_right=[]
        y_left=[]
        y_right=[]

        for i in range(len(x[:])):
            if x[i][attribute_of_vector]<=max_value:
                x_left.append(x[i])
                y_left.append(y[i])
            else:
                x_right.append(x[i])
                y_right.append(y[i])

        return np.array(x_left), np.array(x_right), np.array(y_left),np.array(y_right)

    def information_gain(self, parent, l_child, r_child):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if self.criterion == "gini":
            gain = 1 - (weight_l * self.gini(l_child) + weight_r * self.gini(r_child))
        elif self.criterion == "misclassification":
            gain = 1 - (weight_l * self.misclassification(l_child) + weight_r * self.misclassification(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):

        class_names = np.unique(y)
        entropy = 0
        for cls in class_names:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini(self, y):

        class_names = np.unique(y)
        gini = 0
        for cls in class_names:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def misclassification(self, y):

        class_names = np.unique(y)

        p_cls=[]
        for cls in class_names:
            p_cls.append(len(y[y == cls]) / len(y))

        me=max(p_cls)
        return 1 - me

    def fit(self, X, Y):

        self.root = self.build_tree(X, Y)

    def predict(self, X):

        predictions=[]

        for x in X:
            predictions.append(self._predict(x, self.root))

        return np.array(predictions)

    def _predict(self, x, tree):

        if tree.class_name != None:
            return tree.class_name
        feature_val = x[tree.attribute_of_vector]
        if feature_val <= tree.max_value:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)





digits=load_digits()


x=digits.data

y=digits.target

x,y=shuffle(x, y)

number_of_test=int(0.8*len(x))

x_train=x[:number_of_test]
y_train=y[:number_of_test]

x_test=x[number_of_test:]
y_test=y[number_of_test:]


decisiontree=DecisionTree("misclassification")


decisiontree.fit(x_train, y_train)

trainPrediction=decisiontree.predict(x_train)
print("Train accuarcy: ", accuracy(y_train, trainPrediction))
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_train, trainPrediction)
disp.figure_.suptitle("Confusion Matrix for Training set")




testPrediction=decisiontree.predict(x_test)
print("Test accuarcy: ", accuracy(testPrediction, y_test))
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, testPrediction)
disp.figure_.suptitle("Confusion Matrix for Testing set")
right, wrong=histograms(disp.confusion_matrix)

plt.figure(3)
plt.bar(*(digits.target_names, right))
plt.title("Histogram of right defined digits for testing set")

plt.figure(4)
plt.bar(*(digits.target_names, wrong))
plt.title("Histogram of wrong defined digits for testing set")

plt.show()






