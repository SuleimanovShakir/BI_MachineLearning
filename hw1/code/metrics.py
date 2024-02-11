import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(y_pred.shape[0]):
        if y_pred[i] == '1' and y_true[i] == y_pred[i]:
            TP += 1
        elif y_pred[i] == '0' and y_true[i] == y_pred[i]:
            TN += 1
        elif y_pred[i] == '1' and y_true[i] != y_pred[i]:
            FP += 1
        elif y_pred[i] == '0' and y_true[i] != y_pred[i]:
            FN += 1
        else:
            raise ValueError('Strange number occured')
            
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)

    return precision, recall, f1, accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    counts = 0
    
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_true[i]:
            counts += 1
    accuracy = counts/y_pred.shape[0]
    
    return accuracy

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - (np.sum((np.array(y_true) - y_pred)**2)/np.sum((np.array(y_true) - np.array(y_true).mean())**2))

    return r2
    
def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    mse = np.sum((np.array(y_true) - y_pred)**2)/y_pred.shape[0]

    return mse

def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    
    mae = np.sum(np.abs((np.array(y_true) - y_pred)))/y_pred.shape[0]

    return mae
    