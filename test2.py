import matplotlib.pyplot as plt
import numpy as np

n = 6
m = 5

set1 = np.random.uniform(size=n)
set2 = np.random.uniform(size=n)

gap = 0.01

for j in range(1, 1 + m):
    c = j / (m + 1)
    xvalues1 = [c - i*gap for i in range(1, 1+n)]
    xvalues2 = [c + i*gap for i in range(1, 1+n)]
    plt.bar(xvalues1, set1, width=gap)
    plt.bar(xvalues2, set2, width=gap)
plt.show()

