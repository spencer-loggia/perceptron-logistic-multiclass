import os
import json
import pickle
import argparse as ap

import numpy as np

from MLcode import models

from data import load_data

X, y, nclasses = load_data("/Users/spencerloggia/MachineLearning/MLHW1/data/speech.mc_sm.train")

X = X.toarray();

test_LMC = models.MCLogistic(nfeatures=len(X[0]), nclasses=nclasses)

#print(test_LMC.logits(X))

#print("\n" + str(test_LMC.softmax(test_LMC.logits(X))))

print("\n" + str(test_LMC.predict(X)))

for i in range(0, 100):
    print("\n" + str(test_LMC.fit(X, y, .1)))

print("\n" + str(test_LMC.predict(X)))
