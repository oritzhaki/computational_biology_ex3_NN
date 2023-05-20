import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import main as ga

#get hyperparameters?


df = pd.read_csv('spreadcount.csv')
avg = df.mean(axis=0)
plt.plot(np.arange(1, 501), avg[:500])
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.title(f'Best Fitness In Each Iteration')
plt.savefig('plot.png')
plt.show()
