import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values.

    Returns:
    - The Pearson correlation coefficient between the two columns.
    """
    r = 0.0
    # Calculate the means of x and y
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    # Calculate the Pearson correlation coefficient
    r = np.sum((x - mu_x) * (y - mu_y)) / \
        np.sqrt(np.sum((x - mu_x) ** 2) * np.sum((y - mu_y) ** 2))
    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).
    """
    best_features = []
    # Compute Pearson correlation between each numeric feature in X and label y
    p_correlations = np.abs(np.array([pearson_correlation(X[col], y) for col in X.select_dtypes(include=['number']).columns]))
    # Select top n_features index
    top_index = np.argsort(p_correlations)[-n_features:]
    # Get corresponding column names
    best_features = X.select_dtypes(
        include=['number']).columns[top_index].tolist()
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cost_compute(self, X, y):
        h = self.sigmoid(np.dot(X, self.theta))
        cost = (-1.0 / len(y)) * (np.dot(y.T, np.log(h + 1e-5)
                                         ) + np.dot((1 - y).T, np.log(1 - h + 1e-5)))
        return cost

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        # Bais Trick
        X = np.insert(X, 0, 1, axis=1)
        # Initialize theta
        self.theta = np.random.random(X.shape[1])
        for _ in range(self.n_iter):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y))
            self.theta -= self.eta * gradient

            # Calculate cost and check convergence
            # cost = -y * np.log(h) - (1 - y) * np.log(1 - h)
            # self.Js.append(cost.mean())
            cost = self.cost_compute(X, y)
            self.Js.append(cost)
            self.thetas.append(self.theta.copy())
            if len(self.Js) > 1 and np.abs(cost - self.Js[-2]) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Bais Trick
        X = np.insert(X, 0, 1, axis=1)
        preds = self.sigmoid(np.dot(X, self.theta)).round()
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)


    indices = np.random.permutation(X.shape[0])
    X, y = X[indices], y[indices]

    fold_size = len(y) // folds
    accuracies = []

    for fold in range(folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.vstack((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        model = algo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracies.append((y_pred == y_val).mean())
    cv_accuracy = np.mean(accuracies)
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    # Calculate the exponent part of the formula
    exponent = -((data - mu) ** 2) / (2 * (sigma ** 2))
    # Calculate the denominator part of the formula
    denominator = np.sqrt(2 * np.pi * (sigma ** 2))
    # Calculate the pdf
    p = np.exp(exponent) / denominator
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        n_samples, n_features = data.shape
        self.responsibilities = np.zeros((n_samples, self.k))

        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k

        # Initialize means by randomly selecting k data points
        self.mus = data[np.random.choice(n_samples, self.k, replace=False)].reshape(self.k)
        # self.sigmas = np.random.uniform(3, 1, self.k)
        self.sigmas = np.full(self.k, np.std(data))
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        for i in range(self.k):
            self.responsibilities[:, i] = self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])
        
        sum_responsibilities = np.sum(self.responsibilities, axis=1, keepdims=True)
        self.responsibilities /= sum_responsibilities

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        n_samples = data.shape[0]
        self.weights = np.mean(self.responsibilities, axis=0)
        n_w = n_samples * self.weights

        self.mus = np.dot(self.responsibilities.T, data) / n_w
        for i in range(self.k):
            self.sigmas[i] = np.sqrt(np.inner(self.responsibilities[:, i], (data - self.mus[i]) ** 2) / n_w[i])

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        dataf = data.flatten()

        for _ in range(self.n_iter):
            self.expectation(dataf)
            self.maximization(dataf)
            cost = self.compute_cost(dataf)
            self.costs.append(cost)

            if len(self.costs) > 1 and np.abs(cost - self.costs[-2]) < self.eps:
                break

    def compute_cost(self, data):
        """
        Compute the negative log-likelihood cost function.
        """
        n_samples = data.shape[0]
        log_likelihoods = np.zeros(n_samples)

        for i in range(self.k):
            log_likelihoods += self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i])

        return -np.mean(np.log(log_likelihoods))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    k = len(weights)
    n_pdf = np.array([norm_pdf(data, mus[i], sigmas[i]) for i in range(k)])
    pdf = np.inner(weights, n_pdf.T)
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = None
        self.em_param = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.classes, classes_counts = np.unique(y, return_counts=True)
        num_classes = len(self.classes)
        num_features = X.shape[1]
        num_samples = len(y)
        self.prior = classes_counts / num_samples
        self.em_param = [[] for i in range(num_classes)]

        for i in range(num_classes):
            data_class = X[y == self.classes[i]]
            for j in range(num_features):
                em_ij = EM(k=self.k)
                em_ij.fit(data_class[:,j].reshape(-1, 1))
                self.em_param[i].append((em_ij.get_dist_params()))
                
                
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Number of classes and features
        num_classes = len(self.prior)
        num_features = X.shape[1]
        # Compute Gaussian Mixture Model (GMM) PDFs for each feature of each class
        gmm_pdfs = np.array([[gmm_pdf(X[:, feature_index], *self.em_param[class_index][feature_index]) for feature_index in range(num_features)] for class_index in range(num_classes)])
        # Compute likelihoods by taking the product of GMM PDFs across all features for each class
        likelihoods = np.product(gmm_pdfs, axis=1)
        # Compute posterior probabilities by multiplying likelihoods by the prior probabilities of the classes
        posteriors = likelihoods * self.prior.reshape(-1, 1)
        # Assign each instance to the class with the highest posterior probability
        preds = np.argmax(posteriors, axis=0)
        return preds.reshape(-1, 1)

def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    logistic_regression_model = LogisticRegressionGD(eta= best_eta, eps = best_eps)
    logistic_regression_model.fit(x_train, y_train)
    lor_train_acc = np.mean(logistic_regression_model.predict(x_train)== y_train)
    lor_test_acc = np.mean(logistic_regression_model.predict(x_test)== y_test)
    naive_bayes_gaussian_model = NaiveBayesGaussian(k)
    naive_bayes_gaussian_model.fit(x_train, y_train)
    bayes_train_acc = np.mean(naive_bayes_gaussian_model.predict(x_train)==y_train.reshape(-1, 1))
    bayes_test_acc = np.mean(naive_bayes_gaussian_model.predict(x_test)==y_test.reshape(-1, 1))

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    from scipy.stats import multivariate_normal
    # # Dataset A - Gaussian Naive Bayes is expected to perform better if the dataset
    # # isn't linearly separable
    a_d1 = multivariate_normal.rvs(np.array([4, 4, 4]), np.identity(3)*1.8, size=350)
    a_d2 = multivariate_normal.rvs(np.array([-6, 4, 4]), np.identity(3)*1.2, size=350)
    a_d3 = multivariate_normal.rvs(np.array([18, 18, -8]), np.identity(3)*3.5, size=350)
    a_d4 = multivariate_normal.rvs(np.array([-18, -18, 8]), np.identity(3)*8.5, size=350)
    a_y1 = np.zeros(350)
    a_y2 = np.zeros(350)
    a_y3 = np.ones(350)
    a_y4 = np.ones(350)

    dataset_a_features = np.concatenate((a_d1, a_d2, a_d3, a_d4))
    dataset_a_labels = np.concatenate((a_y1, a_y2, a_y3, a_y4))

    # Dataset B - Linearly separable with high covariance, suitable for Logistic Regression
    d1_cov = np.array([1, 0.7, -0.7,
                      0.7, 1, 0,
                      -0.7, 0, 1.8]).reshape(3, 3)
    d2_cov = np.array([1, 0, 0.7,
                      0, 1, -0.4,
                      0.7, -0.4, 0.9]).reshape(3, 3)
    d3_cov = np.array([1, 0.4, 0,
                      0.4, 1, 0,
                      0, 0, 1]).reshape(3, 3)
    d4_cov = np.array([1, -0.4, 0,
                      -0.4, 1, 0,
                      0, 0, 1]).reshape(3, 3)

    b_d01 = multivariate_normal.rvs(np.array([1, 1, 0]) * 7, d1_cov * 350, size=350)
    b_d02 = multivariate_normal.rvs(np.array([0.8, 1, 0.8]) * 7, d2_cov * 350, size=350)
    b_d03 = multivariate_normal.rvs(np.array([1, 3, 5]) * 7, d3_cov * 350, size=350)
    b_d04 = multivariate_normal.rvs(np.array([0.5, 0.5, 3]) * 7, d4_cov * 350, size=350)

    d2_cov = np.array([1, -0.7, 0.7,
                      -0.7, 1, 0,
                      0.7, 0, 1.8]).reshape(3, 3)
    b_d11 = multivariate_normal.rvs(np.array([1, 1, -6]) * 7, d1_cov * 350, size=350)
    b_d12 = multivariate_normal.rvs(np.array([-3, -5, -9]) * 7, d2_cov * 350, size=350)
    b_d13 = multivariate_normal.rvs(np.array([-9, -9, -6]) * 7, d3_cov * 350, size=350)
    b_d14 = multivariate_normal.rvs(np.array([-9, -9, -9]) * 7, d4_cov * 350, size=350)

    b_y1 = np.zeros(350 * 4)
    b_y2 = np.ones(350 * 4)

    dataset_b_features = np.concatenate((b_d01, b_d02, b_d03, b_d04, b_d11, b_d12, b_d13, b_d14))
    dataset_b_labels = np.concatenate((b_y1, b_y2))
   
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

def plot_3d_dataset(features, labels, title):
    """
    Plot a 3D scatter plot for the dataset.

    Parameters:
    features: array-like, shape = [n_samples, n_features]
      The features of the dataset.
    labels: array-like, shape = [n_samples]
      The labels of the dataset.
    title: str
      The title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap=ListedColormap(['blue','red']))

    # Add labels
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(title)

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    plt.show()
