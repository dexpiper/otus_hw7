import numpy as np
from scipy import sparse
from scipy.special import expit as sigmoid


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            indices_vector = np.random.choice(
                a=num_train,
                size=batch_size,
                replace=True
            )

            X_batch = X[indices_vector,]
            y_batch = y[indices_vector,]

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            m = X_batch.shape[0]
            l = learning_rate
            old_shape = self.w.shape
            self.w = self.w - l*gradW  # HERE
            assert self.w.shape == old_shape

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, X, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################
        prob_class_1 = sigmoid(X * self.w.T)
        prob_class_0 = 1 - prob_class_1
        y_proba = np.c_[prob_class_0, prob_class_1]
        assert y_proba.shape == (X.shape[0], 2), f'y_proba.shape: {y_proba.shape} != {(X.shape[0], 2)}'

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        proba_matrix = self.predict_proba(X, append_bias=True)

        y_pred = np.argmax(proba_matrix, axis=1)
        assert y_pred.shape == (X.shape[0],), f'y_proba.shape: {y_pred.shape} != {(X.shape[0],)}'

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        # print('** Initializing:')
        # print(f'dw_shape: {dw.shape}')
        # print(f'self.w_shape: {self.w.shape}')
        # Compute loss and gradient. Your code should not contain python loops.

        m = X_batch.shape[0]
        # print(f'shape X_batch: {X_batch.shape}\nshape self.w: {self.w.shape}')
        arg = X_batch * self.w.T
        # print(f'Shape of X_batch * self.w.T: {arg.shape}')
        h = sigmoid(arg)
        # print(f'\nShape of h: {h.shape}\nShape of h - y_batch: {(h - y_batch).shape}\n')
        loss = -1/m * np.sum(
            np.dot(np.log(h), y_batch.T) + 
            np.dot(np.log(1 - h), (1 - y_batch.T))
        )
        dw = 1/m * (X_batch.T * (h - y_batch).T)
        assert isinstance(loss, float)
        assert dw.shape == self.w.shape

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.

        loss += (reg/2*m) * np.sum(self.w**2)
        dw = np.append(
            dw[:-1] + (reg/m) * self.w[:-1],
            dw[-1]
        )

        assert isinstance(loss, float)
        assert dw.shape == self.w.shape

        return loss, dw


    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
