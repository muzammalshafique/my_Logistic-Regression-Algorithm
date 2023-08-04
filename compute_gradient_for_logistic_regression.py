import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def cost_logistic(X,y,w,b):
    m=X.shape[0]
    cost = 0
    for i in range(m):
        sig = sigmoid(np.dot(X[i], w) + b)
        cost = cost + (y[i]* math.log(sig) + (1 - y[i]) * math.log (1 - sig))
    cost = cost / (-m)
    return cost

def compute_gradient_logistic(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.
    for i in range(m):
        err_i = sigmoid(np.dot(X[i], w) + b)- y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i
    format
    dj_dw /= m  # (n,)
    dj_db /= m
    return dj_dw, dj_db

# def compute_gradient_logistic(X,Y,w,b):
    m,n = X.shape
    dj_dw_s = []
    dj_db = 0.
    for i in range(n):
        x_s = []
        for j in range(m):
            x = sigmoid(np.dot(X[j],w)+b)-Y[j]
            x_s.append(x)
        y = np.dot(x_s, X[:,i])/m
        dj_db = np.sum(x_s)/m
        dj_dw_s.append(y)
    return dj_dw_s, dj_db

def gradient_descent(X,y,w,b,alpha,num_iters):
    for i in range(num_iters):
        dj_dw,dj_db=compute_gradient_logistic(X,y,w,b)
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db
    return w,b 

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([6.89982915, 6.69858294])
b_tmp = -18.688475083902574
cost = cost_logistic(X_tmp,y_tmp,w_tmp,b_tmp)
print("Thre cost is: ", cost)
dj_dw, dj_db = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_dw: {dj_dw}")
print(f"dj_db: {dj_db}")
alpha = 9.0e-2
num_iters = 30000
w,b=gradient_descent(X_tmp,y_tmp,w_tmp,b_tmp, alpha, num_iters)
print("The optimal values of w and b are: ",w,b)