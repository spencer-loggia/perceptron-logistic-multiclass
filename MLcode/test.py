import os
import json
import pickle
import argparse as ap

import numpy as np

from MLcode import models

from data import load_data

X, y, nclasses = load_data("/Users/spencerloggia/MachineLearning/MLHW1/data/speech.mc.train")

test_LMC = models.MCPerceptron(nfeatures=X.shape[0], nclasses=nclasses)

#print(test_LMC.logits(X))

#print("\n" + str(test_LMC.softmax(test_LMC.logits(X))))

print("\n" + str(test_LMC.predict(X)))

for i in range(0, 100):
    test_LMC.fit(X=X, y=y, lr=.01)

print("\n" + str(test_LMC.predict(X)))
