import numpy as np
import math


def calculate_gaussian_proba(x_mean, x_cov, x_i):
    constant = 1 / (2 * math.pi * math.sqrt(np.linalg.det(x_cov)))
    exponent = - 0.5 * np.mat(x_i - x_mean) * np.linalg.inv(x_cov) * np.mat(x_i - x_mean).T
    return constant * math.exp(exponent)


class BayesDecisionRule:
    def __init__(self):
        self.classes_list = []
        self.x_nums_list = []
        self.x_mean_list = []
        self.x_cov_list = []

    def fit(self, x_train, y_train):
        self.classes_list = np.unique(y_train)
        for i in self.classes_list:
            x = x_train[np.where(y_train == i)]

            x_nums = x.shape[0]  # number of samples
            x_mean = np.mean(x, axis=0)  # mean vector
            x_cov = np.dot((np.mat(x) - x_mean).T, np.mat(x) - x_mean) / x_nums  # variance matrix

            self.x_nums_list.append(x_nums)
            self.x_mean_list.append(x_mean)
            self.x_cov_list.append(x_cov)

    def predict(self, x_test):
        y_test = [0] * len(x_test)
        for j in range(len(x_test)):
            posterior_list = []
            for i in range(len(self.classes_list)):
                prior_i = self.x_nums_list[i] / sum(self.x_nums_list)  # prior probabilities
                cond_i = calculate_gaussian_proba(self.x_mean_list[i], self.x_cov_list[i], x_test[j])  # conditional probability density
                posterior_list.append(prior_i * cond_i)

            y_test[j] = np.argsort(posterior_list)[-1] + 1  # Bayes decision rule
        return y_test

    def score(self, x, label):
        y = self.predict(x)
        return sum(label == y) / len(label)


if __name__ == '__main__':
    path_data_train = r'.\Data_Train.CSV'
    path_label_train = r'.\Label_Train.CSV'
    path_data_test = r'.\Data_Test.CSV'

    data_train = np.loadtxt(path_data_train, delimiter=',')
    data_test = np.loadtxt(path_data_test, delimiter=',')
    label_train = np.loadtxt(path_label_train, delimiter=',')

    clf = BayesDecisionRule()
    clf.fit(data_train, label_train)
    print('score is', clf.score(data_train, label_train))
    print('predict result is', clf.predict(data_test))
