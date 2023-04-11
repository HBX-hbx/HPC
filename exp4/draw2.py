import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

# Read in the data from the text file
dtype = [('block_size', int), ('exec_time', float)]
data = np.loadtxt('res2.txt', dtype=dtype)

# Extract the x, y, and z coordinates from the data
naive = data[::2]
naive = np.sort(naive, order='block_size')
naive = np.array(naive.tolist())

unique, counts = np.unique(naive[:, 0], return_counts=True)
grouped = np.zeros((len(unique), 2))
for i, val in enumerate(unique):
    subset = naive[naive[:, 0] == val, :]
    mean = np.mean(subset[:, 1])
    grouped[i, :] = [val, mean]
naive = grouped

naive_x_line = naive[::, 0]  # block_size
naive_y_line = naive[::, 1]  # exec_time



share = data[1::2]
share = np.sort(share, order='block_size')
share = np.array(share.tolist())

unique, counts = np.unique(share[:, 0], return_counts=True)
grouped = np.zeros((len(unique), 2))
for i, val in enumerate(unique):
    subset = share[share[:, 0] == val, :]
    mean = np.mean(subset[:, 1])
    grouped[i, :] = [val, mean]
share = grouped

share_x_line = share[::, 0]  # block_size
share_y_line = share[::, 1]  # exec_time

embed()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the data
# ax.scatter(naive_x_line, naive_y_line, naive_z_line, label="naive")
# ax.scatter(share_x_line, share_y_line, share_z_line, label="shared_memory")
ax.plot(naive_x_line, naive_y_line, label="naive")
ax.plot(share_x_line, share_y_line, label="shared_memory")

# Set the axis labels
ax.set_xlabel('block_size')
ax.set_ylabel('exec_time')
ax.legend()

# Show the plot
plt.show()
