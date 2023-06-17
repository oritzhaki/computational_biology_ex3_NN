import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import buildnet as ga
import csv

#get hyperparameters?

# 1. lamarckian
#2. darwin
#3. nothing

df = pd.read_csv('spreadcount.csv')
avg = df.mean(axis=0)
with open('avg.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(avg)
# y1 = []
# y2 = []
# y3 = []
# x = np.arange(1, 101)

# with open('avg.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = list(reader)  # Convert the reader to a list
    # y1 = rows[0]
    # y2 = rows[1]
    # y3 = rows[2]
    # y1 = [float(num_str) for num_str in y1]
    # y2 = [float(num_str) for num_str in y2]
    # y3 = [float(num_str) for num_str in y3]

# Create a new figure and axis
# fig, ax = plt.subplots()
#
# # Plot the three graphs
# ax.plot(x, y3, label='Lamarckian')
# ax.plot(x, y2, label='Darwinian')
# ax.plot(x, y1, label='Regular GA')
#
# # Set plot title and labels
# ax.set_title('Three Graphs')
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
#
# # Add a legend
# ax.legend()
#
# # Show the plot
# plt.show()

plt.plot(np.arange(1, 101), avg)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.title(f'Average Fitness In Each Iteration')
plt.savefig('plot.png')
plt.show()
