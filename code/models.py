""" 
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.ws = []
        for w in range(0, nclasses):
            self.ws.append(np.ones(nfeatures, dtype=np.float))
        self.Ws = np.array(self.ws)

    def logits(self, X):
        gs = []
        for w in self.Ws:
            gs.append(np.dot(np.transpose(w), X))
        logits = np.array(gs)
        return logits

    def fit(self, *, X, y, lr):
        # TODO: Implement this!

        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def softmax(self, logits):
        # TODO: Implement this!
        stable_logits = logits - np.max(logits)
        numerator = np.exp(stable_logits)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax
        raise Exception("You must implement this method!")


class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")
