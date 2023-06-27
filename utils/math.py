import torch as tc

def relu(X):
    return tc.where(X>0,X,0)

def relu_dif_1(X):  #* relu g differential1 # X (10,m)
    return tc.where(X>0,1,0)  

def softmax(X):   #* X (10,m)   --> return (10,m)
    max = tc.max(X,axis=0,keepdims=True).values
    e_x = tc.exp(X - max)  #* 減去最大的  讓所有人都是 - 值

    e_sum = tc.sum(e_x,axis=0,keepdims=True)
    f_x = e_x / e_sum

    return f_x

def one_hot_encode(y_label):  #*y (m,)
    m = len(y_label) #m = 50000   
    y_mat = tc.zeros([10,m])
    y_mat[y_label,tc.arange(m)] = 1
    return y_mat