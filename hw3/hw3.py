import numpy as np


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.21,
            (0, 1): 0.09,
            (1, 0): 0.09,
            (1, 1): 0.61
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.35,
            (0, 1): 0.35,
            (1, 0): 0.25,
            (1, 1): 0.05
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.105,
            (0, 0, 1): 0.105,
            (0, 1, 0): 0.075,
            (0, 1, 1): 0.015,
            (1, 0, 0): 0.245,
            (1, 0, 1): 0.245,
            (1, 1, 0): 0.175,
            (1, 1, 1): 0.035,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        X_Y_actual_prob = np.array([X_Y[x, y] for x in X.keys() for y in Y.keys()])
        X_Y_multiplay_prob = np.array([X[x] * Y[y] for x in X.keys() for y in Y.keys()])
        return not np.all(np.isclose(X_Y_actual_prob, X_Y_multiplay_prob))

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        X_Y_C_actual_prob = np.array([X_Y_C[x, y, c] / C[c] for x in X.keys() for y in Y.keys() for c in C.keys()])
        X_Y_C_multiplay_prob = np.array([
            (X_C[x, c] / C[c]) * (Y_C[y, c] / C[c])
            for x in X.keys() for y in Y.keys() for c in C.keys()
        ])
        return np.all(np.isclose(X_Y_C_actual_prob, X_Y_C_multiplay_prob))


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    log_p = k * np.log(rate) - rate - np.sum(np.log(np.arange(1, k + 1)))
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    #computes the sum of the logarithms of the factorials of each observed sample.
    sum_log_factorials = np.sum(np.sum(np.log(np.arange(1, k+1))) for k in samples)
    #Calculate the log-likelihoods:
    likelihoods = -len(samples) * rates - sum_log_factorials + np.log(rates) * np.sum(samples)
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    # Find the rate that maximize the likelihood
    rate = rates[np.argmax(likelihoods)]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    mean = np.mean(samples)
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    # Calculate the exponent part of the formula
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    # Calculate the denominator part of the formula
    denominator = np.sqrt(2 * np.pi * (std ** 2))
    # Calculate the pdf
    p = np.exp(exponent) / denominator
    # p = 1 / (std * np.sqrt(2 * np.pi)) * \
    #     np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = self.dataset[self.dataset[:, -1] == class_value, : -1]
        self.mean = np.mean(self.class_data, axis=0)
        self.std = np.std(self.class_data, axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = len(self.class_data) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = np.product(normal_pdf(x, self.mean, self.std))
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        posterior = likelihood * prior
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    # array whos each value sign if we predict correctly for instance i
    test_set_size = len(test_set)
    correctness_Classified = (test_set[i][-1] == map_classifier.predict(test_set[i][:-1]) for i in range(test_set_size))
    acc = np.sum(correctness_Classified) / test_set_size
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = len(x)
    x_minus_mu = x - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = (2 * np.pi) ** (-d / 2) * det_cov ** (-1 / 2)
    exponent = -0.5 * np.dot(np.dot(x_minus_mu.T, inv_cov), x_minus_mu)
    pdf = norm_const * np.exp(exponent)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = self.dataset[self.dataset[:, -1] == class_value, : -1]
        self.mean = np.mean(self.class_data, axis=0)
        self.cov = np.cov(self.class_data.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = len(self.class_data) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        posterior = likelihood * prior
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_prior()
        posterior1 = self.ccd1.get_prior()
        pred = 0 if posterior0 > posterior1 else 1
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_instance_likelihood(x)
        posterior1 = self.ccd1.get_instance_likelihood(x)
        pred = 0 if posterior0 > posterior1 else 1
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = self.dataset[self.dataset[:, -1] == class_value, : -1]
        self.n_i = len(self.class_data)
        self.num_features = dataset.shape[1] - 1
        self.feature_value_counts = {}

        # Calculate counts for each attribute value given the class
        for j in range(self.num_features):
            feature_values, counts = np.unique(self.class_data[:, j], return_counts=True)
            self.feature_value_counts[j] = dict(zip(feature_values, counts))
        
        # Calculate the number of possible values for each attribute
        self.unique_values = [len(np.unique(dataset[:, j])) for j in range(self.num_features)]

    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        prior = self.n_i / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1
        for j in range(self.num_features):
            x_j = x[j]
            n_ij = self.feature_value_counts[j].get(x_j, 0)
            if n_ij != 0:
                V_j = self.unique_values[j]
                P_xj_given_Ai = (n_ij + 1) / (self.n_i + V_j)
                likelihood *= P_xj_given_Ai
            else:
                likelihood *= EPSILLON
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        posterior = likelihood * prior
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        # array whos each value sign if we predict correctly for instance i
        test_set_size = len(test_set)
        correctness_Classified = (test_set[i][-1] == self.predict(test_set[i][:-1]) for i in range(test_set_size))
        acc = np.sum(correctness_Classified) / test_set_size
        return acc


