# Kalman Filter for Noisy Range Measurements

This repository contains a Jupyter Notebook that demonstrates the use of a **Kalman Filter** to estimate a measured range with noise. The simulation generates noisy measurements around an actual range and applies the Kalman Filter to smooth the readings.

## Overview

### Problem Statement
- We assume an actual range value of **10 meters**.
- However, measurements are subject to **Gaussian noise** with a standard deviation of **0.1**.
- The Kalman Filter is implemented to **reduce noise and estimate the true value** from the noisy measurements.

### Kalman Filter Implementation
The Kalman filter is a recursive algorithm used to estimate the state of a dynamic system. In this implementation:
- We use an **initial guess** for the state.
- A **prediction step** is performed based on the previous estimate.
- A **measurement update step** is applied using the Kalman gain to correct the estimate.

## Installation
To run this notebook, ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib
```

Alternatively, if using Jupyter Notebook, install Jupyter with:

```bash
pip install jupyter
```

## Running the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/Real J/Kalman-Filter-for-Noisy-Range-Measurements.git
   cd Kalman-Filter-for-Noisy-Range-Measurements
   ```
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run the notebook file to see the simulation in action.

## Code Explanation

### 1. Importing Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
```
- `numpy` is used for numerical operations.
- `matplotlib.pyplot` is used for plotting the results.

### 2. Simulating Noisy Measurements
```python
actual_range = 10
num_measurements = 100
measured_range = np.random.normal(actual_range, 0.1, num_measurements)
plt.plot(measured_range, "o-")
```
- The actual range is set to `10` meters.
- `num_measurements` defines the number of sensor readings.
- We generate **Gaussian noise** around the actual range with `np.random.normal()`.
- The noisy measurements are plotted for visualization.

### 3. Initializing Kalman Filter Variables
```python
R = 0.01  # Measurement noise covariance
C = 0.001  # Process noise covariance

x = np.zeros(num_measurements)  # State estimates
p = np.zeros(num_measurements)  # Error covariance
x_minus = np.zeros(num_measurements)
p_minus = np.zeros(num_measurements)
k_gain = np.zeros(num_measurements)

x[0] = 1  # Initial estimate
p[0] = 1  # Initial error covariance
```
- `R` represents measurement noise covariance.
- `C` represents process noise covariance.
- `x` stores the filtered state estimates.
- `p` stores the error covariance values.
- Initial values for `x` and `p` are set.

### 4. Applying the Kalman Filter
```python
for i in range(1, num_measurements):
    x_minus[i] = x[i-1]
    p_minus[i] = p[i-1] + C
    
    k_gain[i] = p_minus[i] / (p_minus[i] + R)
    x[i] = x_minus[i] + k_gain[i] * (measured_range[i] - x_minus[i])
    p[i] = (1 - k_gain[i]) * p_minus[i]
```
- The **prediction step** assumes the next state is the previous estimate (`x_minus[i] = x[i-1]`).
- The **error covariance** is updated (`p_minus[i] = p[i-1] + C`).
- The **Kalman gain** is computed (`k_gain[i]`).
- The **state estimate** is updated using the Kalman equation.
- The **error covariance** is updated.

### 5. Plotting the Results
```python
plt.plot(x)
```
- The Kalman filter output is plotted to visualize the smoothed estimate.

## Results
- The Kalman Filter effectively reduces noise and provides a **smoothed estimate** of the true range.
- The filter adapts over time and converges to the actual value.

![graph](bottle.PNG)


## Applications
- **Sensor fusion**: Used in robotics and autonomous systems.
- **Navigation**: GPS and IMU filtering.
- **Financial modeling**: Noise reduction in stock market data.
- **Biomedical engineering**: ECG and EEG signal processing.

## Contributing
Feel free to contribute by:
- Fixing issues.
- Optimizing the Kalman filter parameters.
- Adding new applications.

## License
This project is open-source under the MIT License.

