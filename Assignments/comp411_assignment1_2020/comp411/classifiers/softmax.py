import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        # calculate loss
        scores = np.dot(X[i], W)

        # shift values to avoid numeric instability
        scores -= scores.max()

        loss += -np.log(
            np.exp(scores[y[i]]) / \
            np.sum((np.exp(scores)))
        )

        # calculate grad
        dW[:, y[i]] += (np.exp(scores[y[i]]) - np.sum(np.exp(scores))) / np.sum(np.exp(scores)) * X[i]
        for j in range(num_classes):
            if j != y[i]:
                dW[:, j] += np.exp(scores[j]) / np.sum(np.exp(scores)) * X[i]

    # average loss and grad over num_train
    loss = loss / num_train
    dW = dW / num_train

    # regularization of loss and grad
    loss += reg_l2 * np.sum(np.square(W))
    dW += 2 * reg_l2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = np.dot(X, W)
    scores = scores - np.max(scores, axis=1, keepdims=True)

    loss = np.sum(
            -np.log(
            ((np.exp(scores)) / \
            np.exp(scores[y]).sum(axis=1, keepdims=True))[np.arange(num_train), y]
        )
    )

    df = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    df[np.arange(num_train), y] -= 1
    dW = np.dot(np.transpose(X), df)

    loss = loss / num_train
    dW = dW / num_train

    loss += reg_l2 * np.sum(np.square(W))
    dW += 2 * reg_l2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
