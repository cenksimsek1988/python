import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets, linear_model
import easygui
# machinelearningmastery.com/machine-learning-in-python-step-by-step/

SCORING_ACCURACY = "accuracy"
KIND_BOX = "box"


class MachineLearning:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        self.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataSet = None
        self.COLUMN_CLASS = "class"
        self.x_train = []
        self.y_train = []
        self.x_validation = []
        self.y_validation = []
        self.array = []
        self.validation_size = None
        self.seed = None
        self.scoring = None
        self.models = []
        self.results = []
        self.names2 = []

    def import_data(self, url):
        # self.dataSet = pandas.read_csv(url, names=self.names)
        self.dataSet = pandas.read_csv(url, header=0)
        self.array = self.dataSet.values

    def shape(self):
        print(self.dataSet.shape)

    def head(self, count):
        print(self.dataSet.head(count))

    def group_count(self, name):
        print(self.dataSet.groupby(name).size())

    def describe(self):
        print(my_teach.dataSet.describe())

    def plot(self, kind):
        self.dataSet.plot(kind=kind, subplots=True, layout=(2, 2), sharex=False, sharey=False)
        plt.show()

    def hist(self):
        self.dataSet.hist()
        plt.show()

    def matrix(self):
        scatter_matrix(self.dataSet)
        plt.show()

    def split(self):
        x = self.array[:, [7, 16]]
        y = self.array[:, 19]
        print(x)
        print(y)
        self.validation_size = 0.20
        self.seed = 7
        self.x_train, self.x_validation, self.y_train, self.y_validation = model_selection.train_test_split(x, y, test_size=self.validation_size, random_state=self.seed)
        self.scoring = SCORING_ACCURACY

    def add_models(self):
        # self.models.append(('LR', LogisticRegression()))
        # self.models.append(('LDA', LinearDiscriminantAnalysis()))
        # self.models.append(('KNN', KNeighborsClassifier()))
        # self.models.append(('CART', DecisionTreeClassifier()))
        # self.models.append(('NB', GaussianNB()))
        # self.models.append(('SVM', SVC()))
        self.models = []
        self.models.append(('LRM', linear_model.LinearRegression()))

    def print_results(self):
        for name, model in self.models:
            model.fit(self.x_train, self.y_train)

            # Make predictions using the testing set
            self.y_validation = model.predict(self.x_validation)

            # The coefficients
            print('Coefficients: \n', model.coef_)
            # The mean squared error
            print("Mean squared error: %.2f"
                  % mean_squared_error(self.y_validation, self.y_validation))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(self.y_validation, self.y_validation))

            # Plot outputs
            plt.scatter(self.x_validation, self.y_validation, color='black')
            plt.plot(self.x_validation, self.y_validation, color='blue', linewidth=3)

            plt.xticks(())
            plt.yticks(())

            plt.show()


            # kfold = model_selection.KFold(n_splits=10, random_state=self.seed)
            # cv_results = model_selection.cross_val_score(model, self.x_train, self.y_train, cv=kfold, scoring=self.scoring)
            # self.results.append(cv_results)
            # self.names2.append(name)
            # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            # print(msg)

    def compare(self):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names2)
        plt.show()

    def predict(self):
        for name, prediction_model in self.models:
            prediction_model.fit(self.x_train, self.y_train)
            predictions = prediction_model.predict(self.x_validation)
            print(name)
            print(accuracy_score(self.y_validation, predictions))
            print(confusion_matrix(self.y_validation, predictions))
            print(classification_report(self.y_validation, predictions))


if __name__ == '__main__':
    my_teach = MachineLearning()
    my_teach.import_data("C:\\Users\\riskactive\oTreeProjects\data2\my_public_goods.csv")
    # my_teach.import_data(easygui.fileopenbox())
    my_teach.shape()
    my_teach.head(20)
    my_teach.describe()
    # my_teach.group_count(my_teach.COLUMN_CLASS)
    # my_teach.plot(KIND_BOX)
    # my_teach.matrix()
    my_teach.split()
    my_teach.add_models()
    my_teach.print_results()
    my_teach.compare()
    my_teach.predict()




