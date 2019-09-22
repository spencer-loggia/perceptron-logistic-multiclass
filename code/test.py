import os
import json
import pickle
import argparse as ap

import numpy as np

from code import models

from data import load_data

test_LMC = models.MCLogistic(nfeatures=10, nclasses=10)

for i in test_LMC.Ws:
        print("")

X, y, nclasses = load_data("/Users/spencerloggia/MachineLearning/MLHW1/data/speech.mc.train")

#print(test_LMC.logits(X))

#print("\n" + str(test_LMC.softmax(test_LMC.logits(X))))

print("\n" + str(test_LMC.predict(X)))