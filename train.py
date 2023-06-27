#%%
from _0_preprocessing import train_loader,val_loader,test_loader #, x_test, y_test
import torch as tc
import gc
class Relu:
    def forward(X):   return tc.where(X>0,X,0)
    def backward(X):  return tc.where(X>0,1,0)  
class Conv:
    def __init__(self,f,p,s,n_C_prev,n_C_now):
       # filter size |  padding  | stride
        self.f, self.p, self.s = f, p, s   
        self.n_C_prev, self.n_C_now = n_C_prev, n_C_now
        beta = 0.1
        self.F = tc.rand(n_C_now,n_C_prev,f,f)*beta*2 - beta    # -range ~ range  # -0.1 ~ 0.1     (0~0.2 - 0.1)
        self.b = tc.rand(1,n_C_now,1,1)       *beta*2 - beta
        self.dF,self.db = tc.zeros_like(self.F),tc.zeros_like(self.b)
    def forward(self,A_prev):
        self.A_prev = A_prev
        self.Z = tc.conv2d(input=A_prev,weight=self.F,stride=self.s,padding=self.p) + self.b
        return self.Z
    def backward(self, dZ):
        self.dF =  tc.conv2d(input = self.A_prev.permute(1,0,2,3), weight = dZ.permute(1,0,2,3), stride = self.s,padding=self.p).permute(1,0,2,3)
        self.db =  tc.sum(dZ,axis = [0,2,3]).reshape(1,self.n_C_now,1,1)   # 合理
        dA_prev = tc.conv_transpose2d(input = dZ,weight = self.F,stride=self.s,padding=self.p) 
        return dA_prev     
class BatchNorm:
    def __init__(self) -> None:
        pass
    def forward(self):
        return
    def backward(self):
        return
class Pool:
    def __init__(self,f,s) -> None:
        self.f, self.s = f, s
        self.pool = tc.nn.MaxPool2d(kernel_size=f,stride=s,padding=0,return_indices=True)
        self.unPool = tc.nn.MaxUnpool2d(kernel_size=f,stride=s,padding=0)
    def forward(self,A_prev):  #!! should be Z_prev
        self.A_prev_size = A_prev.size()
        A, self.A_index = self.pool(A_prev)
        return A
    def backward(self,dA):  
        dA_prev = self.unPool(dA,self.A_index,output_size = self.A_prev_size)
        return dA_prev
class Fc:
    def __init__(self,n_prev,n_now) -> None:
        self.n_prev, self.n_now = n_prev, n_now
        beta = 0.1
        self.W = tc.rand(n_now,n_prev)*beta*2 - beta   # -range ~ range  # -0.1 ~ 0.1
        self.b = tc.rand(n_now,1)     *beta*2 - beta 
        self.dW, self.db = tc.zeros_like(self.W),tc.zeros_like(self.b)
    def forward(self, A_prev):   
        self.A_prev = A_prev
        Z = tc.mm(self.W,A_prev) + self.b
        return Z
    def backward(self,dZ):  
        self.dW = tc.mm(dZ,self.A_prev.T)  # (n_now,m)(m,n_prev) = (n_now,n_prev)
        self.db = tc.sum(dZ,axis = 1).reshape(dZ.shape[0],1)  # (n_now,1)

        dA_prev = tc.mm((self.W).T,dZ)
        return dA_prev   

def forward_prop(X,ls):
    Z0 = ls['conv0'].forward(X)  # 0: conv0
    A0 = Relu.forward(Z0)               
    A1 = ls['pool1'].forward(A0)        # 1: pool0   

    Z2 = ls['conv2'].forward(A1)  # 2: conv1
    A2 = Relu.forward(Z2)
    A3 = ls['pool3'].forward(A2)

    Z4 = ls['conv4'].forward(A3)  # Z0 = F0 A_prev0 + b0
    A4 = Relu.forward(Z4)   
    A5 = ls['pool5'].forward(A4)  # A5   (m, ch_now, f_w, f_h)
    #---------- 後段NN架構 ------------
    m = A5.shape[0]     

    A5_new = A5.reshape(m,-1).T  #(ch_now*f_w*f_h, m)
    Z6 = ls['fc6'].forward(A5_new)
    Aout = tc.nn.functional.softmax(Z6,dim=0)
    return Aout

def backward_prop(Aout,ls,Y_hot):
    m = Y_hot.shape[1]
    #---------- 後段NN架構 ------------
    dZ6 = 1/m * (Aout - Y_hot)
    dA5_new = ls['fc6'].backward(dZ6)   

    dA5 = dA5_new.T.reshape(m,dA5_new.shape[0],1,1)
    
    dA4 = ls['pool5'].backward(dA5)   #dA2(m,16,10,10) 
    dZ4 = dA4*Relu.backward(ls['conv4'].Z)  #dZ2(m,10,11,11)
    ls['conv4'].Z = 0
    dA3 = ls['conv4'].backward(dZ4)

    dA2 = ls['pool3'].backward(dA3)   #dA2(m,16,10,10) 
    dZ2 = dA2*Relu.backward(ls['conv2'].Z)  #dZ2(m,10,11,11)
    ls['conv2'].Z = 0
    dA1 = ls['conv2'].backward(dZ2)

    dA0 = ls['pool1'].backward(dA1)    #dA1(m,6,28,28)
    dZ0 = dA0*Relu.backward(ls['conv0'].Z)  #dZ1(m,6,28,28)
    ls['conv0'].Z = 0
    _   = ls['conv0'].backward(dZ0)      
    return 

def update_params(alpha, ls,m,reg):
    for key in ls:
        if(type(ls[key]) == Fc):
            ls[key].W = ls[key].W*(1-alpha*reg/m) - alpha * ls[key].dW
            ls[key].b = ls[key].b*(1-alpha*reg/m) - alpha * ls[key].db
        elif(type(ls[key]) == Conv):
            ls[key].F = ls[key].F*(1-alpha*reg/m) - alpha * ls[key].dF
            ls[key].b = ls[key].b*(1-alpha*reg/m) - alpha * ls[key].db        
    return

from utils.gpu import gpu_acceleration

loss_list = []
acc_list = {"train":[],"valid":[],"test":[]}
def get_accu(Aout,y_labels,data_type,iteration,acc_list):
    A_pred = tc.argmax(Aout,axis = 0)
    acc_num, tot= tc.sum(A_pred.cpu() == y_labels), len(y_labels)
    print(f"train acc {(acc_num/tot).item()*100:.1f}   {acc_num}/{tot}")
    acc_list[data_type].append([iteration,(acc_num/tot).item()])    
    return

def start_training(train_loader,ls,epochs,lr,reg=3):
    for key in ls:
        if(type(ls[key]) == Fc):  ls[key].W, ls[key].b = gpu_acceleration(ls[key].W, ls[key].b)
        elif(type(ls[key]) == Conv):  ls[key].F, ls[key].b = gpu_acceleration(ls[key].F, ls[key].b)
    iteration = 0
    for epoch in range(epochs):
        for i, (X,y_labels) in enumerate(train_loader):
            # print("iter", i)
            iteration+=1
            m = X.shape[0]
            Y_hot = tc.nn.functional.one_hot(y_labels,num_classes=10).T #Y_hot: (10,m)
            X,Y_hot = gpu_acceleration(X,Y_hot)
        
            Aout = forward_prop(X,ls)
            backward_prop(Aout,ls,Y_hot)
            
            update_params(lr, ls,m,reg)
            if ((epoch % 1) == 0 or (epoch == epochs-1)) and (i%20==0):
                print(f"-----------epoch {epoch}/{epochs}---i {i}/{len(train_loader)}------")
                loss = - tc.sum(Y_hot * tc.log(Aout))/Aout.shape[1]
                # print(f"loss {loss.cpu().item()}")
                loss_list.append([iteration,loss.cpu().item()])

                get_accu(Aout,y_labels,"train",iteration,acc_list)

                X,y_labels = iter(val_loader).next()
                Aout = forward_prop(X.cuda(),ls)
                get_accu(Aout,y_labels,"valid",iteration,acc_list)

                X,y_labels = iter(test_loader).next()
                Aout = forward_prop(X.cuda(),ls)
                get_accu(Aout,y_labels,"test",iteration,acc_list)
                # print(F1.shape,F2.shape)
    return 

if __name__ == "__main__":
    layer_list = {
        'conv0': Conv(f=5,p=2,s=1,n_C_prev=3,n_C_now=32),  #? layer0    (m,3,32,32) --> (m,9,32,32)  
        'pool1': Pool(f=2,s=2),      # layer1    (m,6,32,32) -->  (m,6,16,16)  
        'conv2': Conv(f=5,p=0,s=1,n_C_prev=32,n_C_now=64), #? layer2     (m,6,16,16) -->  (m,16,12,12)            
        'pool3': Pool(f=2,s=2),   # layer3     (m,16,12,12) --> (m,16,6,6)
        'conv4': Conv(f=5,p=0,s=1,n_C_prev=64,n_C_now=128),#layer          (m,16,6,6)  -->  (m,32,2,2)
        'pool5': Pool(f=2,s=2),# layer5    (m,32,2,2) --> (m,32,1,1)
        'fc6'  : Fc(n_prev=128,n_now=10),   #* layer6    (32,m)  --> (10,m)  
    }
    import time, os, pickle 
    start = time.time()     #                         25
    start_training(train_loader,layer_list,epochs=3,lr=0.10,reg=0)
    end = time.time()
    duration =  int(end - start) 
    print("duration",duration)
    

    cnnNetInfo = {
        'file': 'v2_modulization',
        'loss_list': loss_list, 'acc_list':  acc_list,
        'training_time(s)': duration,
        'layerList': layer_list
    }
    outdir = './_train_ML_data/train_W_b_result'
    if not os.path.exists(outdir):  os.makedirs(outdir)
    with open(f"{outdir}/v2_modulization.pkl",'wb') as cnnInfoPickle:
        pickle.dump(cnnNetInfo,cnnInfoPickle)


    print("Piiiiiclke")    


