# Minimax Linear Quadratic Control with Wasserstein Distance
Numerical experiments of Minimax Linear Quadratic Controller using the Wasserstein Metric.

### Prerequisites
* Python 3.7 (Numpy 1.16.0+, Scipy 1.2.1+, and Matplotlib)

### Run
The experimental results in the paper can be reproduced by simply running:
```
python main.py
```

### Input & Parameters
System matrices(A, B, Xi, Q, Q_f, R), the initial state vector(x_0), and the sample data(w) are saved as .npy file in the '/input' folder.
You can modify this data.
Other parameters can be modified by adding commnad line arguments.
For instance, try
```
python main.py --stage_number 200 --test_number 2000 --theta 1.0 --sample_number 20 --sample_mean 0.01 --sample_sigma 0.01

```
If you want to use saved data for sample rather than generating it, try
```
python main.py --stage_number 200 --test_number 2000 --theta 1.0 python main.py --use_saved_sample

```


### Output
All ouptut data and figure will be saved in the '/results' folder after all scripts end.
<img src="Figure_1.gif" alt="drawing" width="400"/>