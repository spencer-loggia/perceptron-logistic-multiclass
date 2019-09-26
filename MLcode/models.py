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

    def scoreOVA(self, Xi):
        return np.dot(self.Ws[1], Xi)


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.Ws = np.zeros((nclasses, nfeatures), dtype=np.float)

    def score(self, Xi):
        dist = []
        for k in range(0, self.nclasses):
            dist.append(np.dot(self.Ws[k], Xi))
        return np.array(dist, dtype=np.float)

    def fit(self, *, X, y, lr):
        X = self._fix_test_feats(X)
        X = X.toarray()
        for i in range(0, len(X)):
            yhat_i = np.argmax(self.score(X[i]))
            if yhat_i != y[i]:
                Ws = self.Ws
                self.Ws[yhat_i] = self.Ws[yhat_i] - lr * X[i]
                self.Ws[y[i]] = self.Ws[y[i]] + lr * X[i]

    def predict(self, X):
        X = self._fix_test_feats(X)
        if type(X) != np.ndarray:
            X = X.toarray()
        predictions = []
        for i in range(0, len(X)):
            dist = self.score(X[i])
            predictions.append(np.argmax(dist))
        return predictions


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        self.nclasses = nclasses
        self.nfeatures = nfeatures
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.Ws = np.zeros((nclasses, nfeatures), dtype=np.float)

    def logits(self, X):
        gs = []
        for w in self.Ws:
            gs.append(np.dot(np.transpose(w), X))
        logits = np.array(gs)
        return logits

    def fit(self, X, y, lr):
        X = self._fix_test_feats(X)
        X = X.toarray()
        for i in range(0, len(X)):
            probs = self.softmax(self.logits(X[i]))
            for k in range(0, self.nclasses):
                if k == y[i]:
                    grad = X[i] - (probs[k] * X[i])
                else:
                    grad = 0 - probs[k] * X[i]
                self.Ws[k] = self.Ws[k] + lr*grad

    def predict(self, X):
        X = self._fix_test_feats(X)
        X = X.toarray()
        predictions = []
        for i in range(0, len(X)):
            y = []
            for k in range(0, self.nclasses):
                y.append(np.dot(self.Ws[k], X[i]))
            predictions.append(np.argmax(y))
        return predictions

    def softmax(self, logits):
        stable_logits = logits - np.max(logits)
        numerator = np.exp(stable_logits)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax



class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        for k in range(0, len(self.models)):
            # build new labels
            new_y = []
            for i in y:
                if i == k:
                    new_y.append(1)
                else:
                    new_y.append(0)
            self.models[k].fit(X=X, y=new_y, lr=lr)

    def predict(self, X):
        X = self._fix_test_feats(X)
        X = X.toarray()
        predictions = []
        for i in range(0, X.shape[0]):
            scores = []
            for k in range(0, self.num_classes):
                scores.append(self.models[k].scoreOVA(X[i]))
            predictions.append(np.argmax(scores))
        return predictions


