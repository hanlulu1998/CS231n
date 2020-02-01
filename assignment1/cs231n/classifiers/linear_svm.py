import numpy as np
from random import shuffle


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
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

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
            margin = scores[j] - correct_class_score + 1  # 设置delta = 1
            if margin > 0:
                loss += margin
                # 可以简单推导出max(0，X_i·W_i-X_i·W_yi)的梯度
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # SVM损失函数L=1/N*sum(max(0，X_i·W_i-X_i·W_yi))+reg*|W|^2
    # 取平均值
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    # 这里乘以0.5是为了Dw求梯度后为0.5*2*reg*W，方便取整
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)
    margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1
    margin[np.arange(num_train), y] = 0.0  # j！=yi 不计算
    margin = (margin > 0) * margin
    loss += margin.sum()
    # 取平均值
    loss /= num_train
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # 向量化的梯度推导，所有j！=yi的元素贡献一次+X_i，所有j=yi的元素献所有max>0个数次-X_i
    margin = (margin > 0) * 1
    row_sum = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] = -row_sum
    dW = X.T.dot(margin)
    # 取平均值
    dW /= num_train
    # 正则化
    dW += reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
