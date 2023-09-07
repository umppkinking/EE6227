import numpy as np


class FisherDiscriminantAnalysis:
    def __init__(self):
        self.nums_features = None  # the number of features
        self.nums_classes = None  # the number of classes
        self.m_i_list = []  # mean vectors
        self.direction_vector_list = []  # direction vectors
        self.bias_list = []  # biases
        self.max_m_i_map_list = []  # the indexes of maximum projection of each mapping

    def calculate_eigen(self, x_train, y_train):
        self.nums_features = x_train.shape[1]
        means = np.mean(x_train, axis=0)
        classes = np.unique(y_train)
        self.nums_classes = classes.shape[0]
        class_list = []
        self.m_i_list = []
        s_i_list = []
        s_w = np.zeros((self.nums_features, self.nums_features))  # within-class scatter matrix

        for i in classes:
            x = x_train[np.where(y_train == i)]
            class_list.append(x)
            m_i = np.mean(x, axis=0)
            self.m_i_list.append(m_i)
            s_i = np.dot((np.mat(x) - m_i).T, np.mat(x) - m_i)
            s_i_list.append(s_i)
            s_w += s_i

        s_t = np.dot((np.mat(x_train) - means).T, np.mat(x_train) - means)  # total scatter matrix
        s_b = s_t - s_w  # between-class scatter matrix
        s_w_i = np.linalg.inv(s_w)
        eigen_value, eigen_vector = np.linalg.eig(np.dot(s_w_i, s_b))
        return eigen_value, eigen_vector

    def fit(self, x_train, y_train):
        eigen_value, eigen_vector = self.calculate_eigen(x_train, y_train)

        for i in range(1, self.nums_classes):
            max_i_eigen_value_index = np.argsort(eigen_value)[-i]
            self.direction_vector_list.append(eigen_vector[max_i_eigen_value_index])  # direction vectors
            m_i_map_list = []
            for j in self.m_i_list:
                m_i_map_list.append(float(np.dot(eigen_vector[max_i_eigen_value_index], j)))
            self.max_m_i_map_list.append(np.argsort(m_i_map_list)[-1] + 1)  # the indexes of maximum projection of each mapping
            m_i_map_list.sort(reverse=True)
            w_i = - (m_i_map_list[0] + m_i_map_list[1]) / 2
            self.bias_list.append(w_i)  # biases
        return

    def predict(self, x_test):
        nums_test = x_test.shape[0]
        g_list = np.dot(x_test, self.direction_vector_list[0].T) + self.bias_list[0]  # discriminant functions
        for i in range(1, self.nums_classes - 1):
            g_list = np.concatenate((g_list, np.dot(x_test, self.direction_vector_list[i].T) + self.bias_list[i]), axis=1)

        y_test = [0] * nums_test
        for i in range(nums_test):
            max_i = np.argsort(g_list[i])[0, -1]
            if g_list[i][0, max_i] > 0:  # decision rules
                y_test[i] = self.max_m_i_map_list[max_i]
            else:
                y_test[i] = self.nums_classes

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

    clf = FisherDiscriminantAnalysis()
    clf.fit(data_train, label_train)
    print('score is', clf.score(data_train, label_train))
    print('predict result is', clf.predict(data_test))
