import numpy as np
import pandas as pd

data = pd.DataFrame(np.arange(16).reshape(4, 4), index=list('abcd'), columns=list('ABCD'))
var = data.loc["a", :]
print(var)
var2 = data.loc[:, "A"]
print(var2)


