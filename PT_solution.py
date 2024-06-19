import matplotlib.pyplot as plt
import numpy as np

# Define the source and destination points
src_points = np.array([[50, 100], [450, 300], [250, 500], [100, 400]])
dst_points = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])

# Number of points
n_points = src_points.shape[0]

# Create the matrix A and vector b for the linear equations Ax = b
A = np.zeros((2 * n_points, 8))
b = np.zeros(2 * n_points)

for i in range(n_points):
    x, y = src_points[i, 0], src_points[i, 1]
    x_prime, y_prime = dst_points[i, 0], dst_points[i, 1]
    
    # Equations for the x coordinates
    A[2*i] = [x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime]
    b[2*i] = x_prime
    
    # Equations for the y coordinates
    A[2*i+1] = [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]
    b[2*i+1] = y_prime

# Solve the system of linear equations for the transformation matrix
m = np.linalg.lstsq(A, b, rcond=None)[0]
m = np.append(m, 1)  # Append 1 for m33 to form the 3x3 matrix

# Reshape into 3x3 matrix
M = m.reshape((3, 3))


# Apply the transformation matrix M to the src_points
transformed_points = np.dot(M, np.vstack((src_points.T, np.ones(n_points))))

# Normalize the points
transformed_points /= transformed_points[2,:]

# Plotting the original and transformed points
fig, ax = plt.subplots(figsize=(8, 6))

# Plot original points
ax.scatter(src_points[:, 0], src_points[:, 1], color='blue', label='Original Points')
ax.plot(src_points[[0, 1, 2, 3, 0], 0], src_points[[0, 1, 2, 3, 0], 1], 'b--', label='Original Quadrilateral')

# Plot transformed points
ax.scatter(transformed_points[0, :], transformed_points[1, :], color='red', label='Transformed Points')
ax.plot(transformed_points[0, [0, 1, 2, 3, 0]], transformed_points[1, [0, 1, 2, 3, 0]], 'r-', label='Transformed Quadrilateral')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
ax.set_title('Original and Transformed Points Visualization')
plt.grid(True)
plt.show()
