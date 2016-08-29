import numpy as np
import scipy.stats
import pandas

'''#######################################Welch's t-Test Exercise###################################'''


def compare_averages(filename):
    """
    Performs a t-test on two sets of baseball data (left-handed and right-handed hitters).

    You will be given a csv file that has three columns.  A player's
    name, handedness (L for lefthanded or R for righthanded) and their
    career batting average (called 'avg'). You can look at the csv
    file by downloading the baseball_stats file from Downloadables below.

    Write a function that will read that the csv file into a pandas data frame,
    and run Welch's t-test on the two cohorts defined by handedness.

    One cohort should be a data frame of right-handed batters. And the other
    cohort should be a data frame of left-handed batters.

    We have included the scipy.stats library to help you write
    or implement Welch's t-test:
    http://docs.scipy.org/doc/scipy/reference/stats.html

    With a significance level of 95%, if there is no difference
    between the two cohorts, return a tuple consisting of
    True, and then the tuple returned by scipy.stats.ttest.

    If there is a difference, return a tuple consisting of
    False, and then the tuple returned by scipy.stats.ttest.

    For example, the tuple that you return may look like:
    (True, (9.93570222, 0.000023))
    """
    # Your code is here:
    df = pandas.read_csv(filename)
    list1 = df[df['handedness'] == 'R']['avg']
    list2 = df[df['handedness'] == 'L']['avg']
    result = scipy.stats.ttest_ind(list1, list2, equal_var=False)
    if result[1] <= 0.05:
        return False, result
    else:
        return True, result


'''####################################Gradient Descent in Python###################################'''


def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features
    (input data points) and values (output data points).
    """
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2 * m)

    return cost


def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # Write code here that performs num_iterations updates to the elements of theta.
    # times. Every time you compute the cost for a given list of thetas, append it
    # to cost_history.
    # See the Instructor notes for hints.

    cost_history = []

    # YOUR CODE GOES HERE
    m = len(values)
    for i in range(num_iterations):
        transpose = np.dot((values - np.dot(features, theta)), features)
        theta += alpha / m * transpose
        cost_history.append(compute_cost(features, values, theta))
        return theta, pandas.Series(cost_history)  # leave this line for the grader

'''#################################Calculating R^2###################################################'''


def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced
    # predictions.
    #
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE
    r_squared = 1 - np.square(data - predictions).sum()/np.square(data - np.mean(data)).sum()
    # Alternatively, you could also do sth. like this:
    # SST = (data - np.mean(data)**2).sum()
    # SSReg = ((predictions - data)**2).sum()
    # r_squared = 1 - SSReg/SST
    return r_squared
