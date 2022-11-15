import csv
import matplotlib.pyplot as plt
import numpy as np

y = np.array([])

with open('rewards.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        y = np.append(y, float(row[0]))

x = np.arange(start=30, stop=30*(y.shape[0]+1), step=30)

plt.plot(x, y)
plt.show()