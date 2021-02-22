# Minimax Linear Quadratic Control with Wasserstein Distance
Numerical experiments of Minimax Linear Quadratic Controller using the Wasserstein Metric.

### Prerequisites
* Python 3.7 (Numpy 1.16.0+, Scipy 1.2.1+, and Matplotlib)

### Run
The experimental results in the paper can be reproduced by running:
```
python main.py
```

### Input & Parameters
System matrices(A, B, Xi, Q, Q_f, R), the initial state(x_0), and the sample(w) are saved as npy file in '/input' folder.
You can modify this data.


### Output
<img src="Figure_1.gif" alt="drawing" width="400"/>