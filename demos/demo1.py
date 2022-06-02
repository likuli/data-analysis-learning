import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

s = pd.Series(np.random.rand(5))
print(s)
print(type(s))

print(s.index, type(s.index))
print(s.values, type(s.values))

print(pd.__version__)




