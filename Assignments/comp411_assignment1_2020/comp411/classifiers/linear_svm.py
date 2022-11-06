import numpy as np
from random import shuffle
import builtins


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
  
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += - X[i] 
                dW[:,j] += X[i] 
            else:
                dW[:,j] += np.zeros(dW[:,j].shape) 
 
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # Following lines are added:
            # dW[:,y[i]] += - X[i] 
            #    dW[:,j] += X[i] 
            # lse:
            #    dW[:,j] += np.zeros(dW[:,j].shape) 
            # 
            # dW /= num_train
            # dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW

def huber_loss_naive(W, X, y, reg):
    """
    Modified Huber loss function, naive implementation (with loops).
    Delta in the original loss function definition is set as 1.
    Modified Huber loss is almost exactly the same with the "Hinge loss" that you have 
    implemented under the name svm_loss_naive. You can refer to the Wikipedia page:
    https://en.wikipedia.org/wiki/Huber_loss for a mathematical discription.
    Please see "Variant for classification" content.
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    ###############################################################################
    # TODO:                                                                       #
    # Complete the naive implementation of the Huber Loss, calculate the gradient #
    # of the loss function and store it dW. This should be really similar to      #
    # the svm loss naive implementation with subtle differences.                  #
    ###############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    delta = 1
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score  
            if margin < - delta:
                dW[:,j] += np.zeros(dW[:,j].shape) 

            elif margin <= delta:
                margin += delta
                loss += margin ** 2
                dW[:,y[i]] += - 2 * X[i] * margin 
                dW[:,j] += 2 * X[i] * margin
            else:
                loss += 4 * margin
                dW[:,y[i]] += - 4 * X[i] 
                dW[:,j] +=  4 * X[i]
                            
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    delta = 1

    scores = X.dot(W)
    correct_score_indexes = np.arange(num_train),y # for each row get y_col 
    correct_scores = np.reshape(scores[correct_score_indexes],(num_train,1)) 
    margin = scores - correct_scores + delta
    margin[correct_score_indexes] = 0 
    margin = np.maximum(0, margin)
     
    loss = np.sum(margin) / num_train + reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    contribution_coefs = np.asarray(margin > 0, dtype=int) # 1's 0's from hinge loss router
    contribution_coefs[correct_score_indexes] = - np.sum(contribution_coefs, axis=1) # negative contributions of correct class
    
    dW = np.matmul(X.T,contribution_coefs) / num_train + reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def huber_loss_vectorized(W, X, y, reg):
    """
    Structured Huber loss function, vectorized implementation.

    Inputs and outputs are the same as huber_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured Huber loss, storing the  #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Huber loss
    # 1. margin < - delta          := zero
    # 2. -delta <= margin <= delta := (margin + 1) ^ 2
    # 3. ow.                       := 4margin

    num_train = X.shape[0]
    delta = 1

    scores = X.dot(W)  
    correct_score_indexes = np.arange(num_train),y 
    correct_scores = np.reshape(scores[correct_score_indexes],(num_train, 1)) 
    margin = scores - correct_scores
    
    lossExtention = np.copy(margin)
    
    firstCond = margin < - delta
    lossExtention[firstCond] = 0 
    
    secondCond = (-delta <= margin) & (margin <= delta)
    lossExtention[secondCond] += delta
    lossExtention[secondCond] **= 2
    
    thirdCond = margin > delta
    lossExtention[thirdCond] *= 4
    
    lossExtention[correct_score_indexes] = 0 
    
    loss = np.sum(lossExtention) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    contribution_coefs = np.zeros(margin.shape)
    
    contribution_coefs[firstCond] = 0
    contribution_coefs[secondCond] = (margin[secondCond] + 1) * 2
    contribution_coefs[thirdCond] = 4
    contribution_coefs[correct_score_indexes] = 0

    contribution_coefs[correct_score_indexes] = - np.sum(contribution_coefs, axis=1) 
  
    dW = np.matmul(X.T,contribution_coefs) / num_train + reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
