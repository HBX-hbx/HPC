import matplotlib.pyplot as plt
import numpy as np

# Read in the data from the text file
data = np.loadtxt('res.txt')

# Extract the x, y, and z coordinates from the data
naive_x_line = data[::2, 0]  # block_size_x
naive_y_line = data[::2, 1]  # block_size_y
naive_z_line = data[::2, 2]  # exec_time

share_x_line = data[1::2, 0]  # block_size_x
share_y_line = data[1::2, 1]  # block_size_y
share_z_line = data[1::2, 2]  # exec_time

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
# ax.scatter(naive_x_line, naive_y_line, naive_z_line, label="naive")
# ax.scatter(share_x_line, share_y_line, share_z_line, label="shared_memory")
ax.plot(naive_x_line, naive_y_line, naive_z_line, label="naive")
ax.plot(share_x_line, share_y_line, share_z_line, label="shared_memory")

# Set the axis labels
ax.set_xlabel('block_size_x')
ax.set_ylabel('block_size_y')
ax.set_zlabel('exec_time')
ax.legend()

# Show the plot
plt.show()
