import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


X = [1, 2, 3]

h = 0
C = 0


Wf = 0.5   
Wi = 0.6   
Wc = 0.7   
Wo = 0.8  

print("Initial Hidden State:", h)
print("Initial Cell State:", C)
print("\n")


for t, x in enumerate(X):

    print("Time Step:", t+1)
    print("Input:", x)

    f = sigmoid(Wf * (h + x))
    print("Forget Gate:", f)

    i = sigmoid(Wi * (h + x))
    print("Input Gate:", i)


    C_tilde = tanh(Wc * (h + x))
    print("Candidate Cell State:", C_tilde)

    C = f * C + i * C_tilde
    print("Updated Cell State:", C)

 
    o = sigmoid(Wo * (h + x))
    print("Output Gate:", o)


    h = o * tanh(C)
    print("Hidden State:", h)

    print("                           ")
 


Wy = 4
b = 0.2

y_hat = Wy * h + b

print("\nFinal Prediction:", y_hat)
