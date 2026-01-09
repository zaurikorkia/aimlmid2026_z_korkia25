import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Given data points
x = np.array([-9, -7, -5, -3.5, -1, 1, 3, 5, 7.9, 9.9])
y = np.array([4, 4.5, 3, 4.5, 1, 1.3, -2, -3.6, -4.9, -5.8])

# Create DataFrame
df = pd.DataFrame({"x": x, "y": y})

# Pearson correlation coefficient
r = df["x"].corr(df["y"])
print("Pearson correlation coefficient (R):", r)

# Regression line (y = mx + b)
m, b = np.polyfit(x, y, 1)
y_line = m * x + b

# Plot
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_line, color="red", label="Regression line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Correlation with Regression Line")
plt.legend()
plt.grid(True)
plt.show()
