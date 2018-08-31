Instruction for running the code:

Please note that the data has not been saved on git and needs to be added in the AwA_python/ folder with the prescribed names.

1. Method 1: Run the following command in the terminal with a virtual environment (virtualenv) activated. 
```
python convex.py
```
It will print the result in the following form: 'Test accuracy for prototype based classification with method 1: 0.xxxxxxxxx'.

2. Method 2: Run the following command in the terminal with a virtual environment (virtualenv) activated. 
```
python regress.py
```
It will print the result in the following form: 'Test accuracy for prototype based classification with method 2 with lambda = xxx: 0.xxxxxxxxx' for each hyperparameter lambda in [0.01, 0.1, 1, 10, 20, 50, 100].

