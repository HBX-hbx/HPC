import matplotlib.pyplot as plt
import numpy as np

g_path = 'output_g.txt'
s_path = 'output_s.txt'

# data = np.loadtxt(g_path)
# x = data[:, 0]
# y = data[:, 1]

# plt.xlabel('stride')
# plt.ylabel('bandwidth')
# plt.plot(x, y)
# plt.show()

data = np.loadtxt(s_path)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
# ax.scatter(naive_x_line, naive_y_line, naive_z_line, label="naive")
# ax.scatter(share_x_line, share_y_line, share_z_line, label="shared_memory")
ax.scatter(x, y, z)

# Set the axis labels
ax.set_xlabel('stride')
ax.set_ylabel('bit width')
ax.set_zlabel('bandwidth')
ax.legend()

# Show the plot
plt.show()
