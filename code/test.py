import os
import json
import pickle
import argparse as ap

import numpy as np

from code import models

from code import data

test_LMC = models.MCLogistic(nfeatures=10, nclasses=10)

for i in test_LMC.Ws:
        print(i)

X = [1, 1, 1, 1, 1, 5, 1, 1, 1, 1]

print(test_LMC.logits(X))

print(test_LMC.softmax(test_LMC.logits(X)))
