import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss=[]
  #dw = []
                
  scores = np.dot(X,W)
  prob = np.zeros_like(scores)
  (N,C) = scores.shape
  
  for j in range(N):
      scores[j]-=np.max(scores[j])
      for k in range(C):
          prob[j][k] = np.exp(scores[j][k]) / np.sum(np.exp(scores[j]))
      yj = y[j]
      loss_i = -np.log(prob[j][yj])
      loss += loss_i
      prob[j][yj] -=1
  prob /= N
  loss = loss/N + 0.5*reg*np.sum(W*W)
  dW = np.dot(X.T,prob) + reg*W
               
                

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

    

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss = []
  #dW = []
  
  N = X.shape[0]
  scores = np.dot(X,W)
  scores -= np.max(scores,axis=1,keepdims=True)
  prob = np.exp(scores)
  prob /= np.sum(prob, axis=1, keepdims=True)
  loss = np.sum(-np.log(prob[np.arange(N),y]))/N + 0.5*reg*np.sum(W*W)
  prob[np.arange(N),y]-=1
  prob/=N
  dW = np.dot(X.T,prob)+reg*W    
     
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

