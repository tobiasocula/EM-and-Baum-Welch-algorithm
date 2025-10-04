import numpy as np

a = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

print(np.argmax(a, axis=1))

a = [0, 0, 1, 1, 0, 1, 1]
b = [1, 0, 1, 0, 1, 1, 1]

nonz = np.count_nonzero(np.logical_xor(a, b))
n = len(a)
score = 1 - nonz / n
print(score)

import matplotlib.pyplot as plt


plt.bar(np.linspace(0, 1, 6), [1, 2, 3, 4, 5, 6], width=1/6)
plt.show()