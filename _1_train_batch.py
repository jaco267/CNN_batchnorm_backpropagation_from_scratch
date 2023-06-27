#%%
from _0_preprocessing import train_loader,val_loader,test_loader #, x_test, y_test
import torch as tc
import gc

class Relu:
    def forward(X):   return tc.where(X>0,X,0)
    def backward(X):  return tc.where(X>0,1,0)  
class Conv:
    def __init__(self,f,p,s,n_C_prev,n_C_now):
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
       # filter size |  padding  | stride
        self.f, self.p, self.s = f, p, s   
        self.n_C_prev, self.n_C_now = n_C_prev, n_C_now
        beta = 0.001
        self.F = tc.rand(n_C_now,n_C_prev,f,f,device=device)*beta*2 - beta    # -range ~ range  # -0.1 ~ 0.1     (0~0.2 - 0.1)
        self.dF= tc.zeros_like(self.F,device=device)
        self.V_F = tc.zeros_like(self.F,device=device)  # momentum
    def forward(self,A_prev):
        self.A_prev = A_prev
        self.Z = tc.conv2d(input=A_prev,weight=self.F,stride=self.s,padding=self.p) #+ self.b
        return self.Z
    def backward(self, dZ):
        self.dF =  tc.conv2d(input = self.A_prev.permute(1,0,2,3), weight = dZ.permute(1,0,2,3), stride = self.s,padding=self.p).permute(1,0,2,3)
        dA_prev = tc.conv_transpose2d(input = dZ,weight = self.F,stride=self.s,padding=self.p) 
        return dA_prev     
class BatchNorm:
    def __init__(self,n_C) -> None:
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.device= device
        self.ini = True
        self.gama = tc.ones((1,n_C,1,1),dtype=tc.float32,device=device)
        self.beta = tc.zeros((1,n_C,1,1),dtype=tc.float32,device=device)
        # self.running_mean_x = tc.zeros((1,n_C,1,1),dtype=tc.float32,device=device)  #note: you can also set it's initial value to self.mean_x
        # self.running_var_x  = tc.zeros((1,n_C,1,1),dtype=tc.float32,device=device)
    def update_running_variables(self,f_num,x):
        if self.ini == True:   # todo  move it to __init__  running_mean_x = tc.zeros(1,nc,1,1)
            print("ini true")
            self.running_mean_x, self.running_var_x = self.mean_x, self.var_x
            self.ini = False
        else:
            alpha = 0.9
            self.running_mean_x = alpha*self.running_mean_x + (1.0-alpha)*self.mean_x
            self.running_var_x  = alpha*self.running_var_x  + (1.0-alpha)*self.var_x
    def forward(self, x ,train:bool):  # x (m,16,32,32)
        mm =  x.shape[0]*x.shape[2]*x.shape[3]
        self.mm = mm
        if train:
            #            1/mm*(x).sum
            self.mean_x = (x).mean([0,2,3],keepdim=True)                #*1,16,1,1
            #            1/mm*((x-self.mean_x)**2).sum
            self.var_x  = (x).var([0,2,3],unbiased=False,keepdim=True)  #*1,16,1,1
            self.update_running_variables(x.shape[1],x)
        else:
            self.mean_x, self.var_x = self.running_mean_x, self.running_var_x
        eps = 0.001
        self.var_x += eps
        # print("max x ",tc.max(x))
        self.x_minus_mean = x - self.mean_x
        self.x_hat = self.x_minus_mean / (self.var_x**(0.5))  #todo self.var_x ** (1/2)
        # print("max x_hat",tc.max(self.x_hat))
        y = self.gama * self.x_hat + self.beta
        return  y
    def backward(self,dy):
        mm = self.mm  #
        self.dgama =  (dy*self.x_hat).sum([0,2,3],keepdim=True)   # dgama for each channel
        self.dbeta =  (dy).sum([0,2,3],keepdim=True)              # dbias for each channel
        dx_hat = dy*self.gama
        # 1, ch, 1, 1
        std = tc.sqrt(self.var_x)
        
        #                     (m,16,w,h)                                       (1,16,1,1)
        dvar_x = (-0.5*dx_hat*(self.x_minus_mean)).sum((0,2,3),keepdim=True) * (self.var_x**-1.5)
        dmean_x = ( dx_hat * (-1.0/std) ).sum((0,2,3),keepdim=True)+ \
                  dvar_x*(-2.0*self.x_minus_mean).sum((0,2,3),keepdim=True)/mm

        dx = dx_hat / std + dmean_x/mm +\
             dvar_x*(2/mm)*self.x_minus_mean
        return  dx
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
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.n_prev, self.n_now = n_prev, n_now
        beta = 0.01
        self.W = tc.rand(n_now,n_prev,device=device)*beta*2 - beta   # -range ~ range  # -0.1 ~ 0.1
        self.b = tc.rand(n_now,1,device=device)     *beta*2 - beta 
        self.dW, self.db = tc.zeros_like(self.W,device=device),tc.zeros_like(self.b,device=device)
        self.V_W,self.V_b = tc.zeros_like(self.W,device=device),tc.zeros_like(self.b,device=device)
    def forward(self, A_prev):   
        self.A_prev = A_prev
        Z = tc.mm(self.W,A_prev) + self.b
        return Z
    def backward(self,dZ):  
        self.dW = tc.mm(dZ,self.A_prev.T)  # (n_now,m)(m,n_prev) = (n_now,n_prev)
        self.db = tc.sum(dZ,axis = 1).reshape(dZ.shape[0],1)  # (n_now,1)

        dA_prev = tc.mm((self.W).T,dZ)
        return dA_prev   

def forward_prop(X,ls,train=True):
    X = ls['conv0'].forward(X)  # 0: conv0   # Z0 (m,3,32,32)
    X = ls['bn0'].forward(X,train)
    X = Relu.forward(X)               
    X = ls['pool1'].forward(X)        # 1: pool0   

    X = ls['conv2'].forward(X)  # 2: conv1
    X = ls['bn2'].forward(X,train)
    X = Relu.forward(X)
    X = ls['pool3'].forward(X)

    X = ls['conv4'].forward(X)  # Z0 = F0 A_prev0 + b0
    X = ls['bn4'].forward(X,train)
    X = Relu.forward(X)   
    X = ls['pool5'].forward(X)  # A5   (m, ch_now, f_w, f_h)

    # X = ls['conv6'].forward(X)  # Z0 = F0 A_prev0 + b0
    # X = ls['bn6'].forward(X,train)
    # X = Relu.forward(X)   
    # X = ls['pool7'].forward(X)  # A5   (m, ch_now, f_w, f_h)
    #---------- 後段NN架構 ------------
    m = X.shape[0]     
    last_shape = X.shape
    X = X.reshape(m,-1).T  #(ch_now*f_w*f_h, m)
    X = ls['fc0'].forward(X)
    Aout = tc.nn.functional.softmax(X,dim=0)
    gc.collect()
    return Aout,last_shape

def backward_prop(Aout,ls,Y_hot,last_shape):
    m = Y_hot.shape[1]
    #---------- 後段NN架構 ------------
    dX = 1/m * (Aout - Y_hot)
    dX = ls['fc0'].backward(dX)   
    dX = dX.T.reshape(m,last_shape[1],last_shape[2],last_shape[3])

    # dX = ls['pool7'].backward(dX)   #dA2(m,16,10,10) 
    # dX = dX*Relu.backward(ls['conv6'].Z)  #dZ2(m,10,11,11)
    # dX = ls['bn6'].backward(dX)
    # dX = ls['conv6'].backward(dX)

    dX = ls['pool5'].backward(dX)   #dA2(m,16,10,10) 
    dX = dX*Relu.backward(ls['conv4'].Z)  #dZ2(m,10,11,11)
    dX = ls['bn4'].backward(dX)
    dX = ls['conv4'].backward(dX)

    dX = ls['pool3'].backward(dX)   #dA2(m,16,10,10) 
    dX = dX*Relu.backward(ls['conv2'].Z)  #dZ2(m,10,11,11)
    dX = ls['bn2'].backward(dX)
    dX = ls['conv2'].backward(dX)

    dX = ls['pool1'].backward(dX)    #dA1(m,6,28,28)
    dX = dX*Relu.backward(ls['conv0'].Z)  #dZ1(m,6,28,28)
    dX = ls['bn0'].backward(dX)
    _   = ls['conv0'].backward(dX)    
    gc.collect()
    return 

def update_params(lr, ls,m,reg):
    beta = 0.9
    for key in ls:
        if(type(ls[key]) == Fc):
            ls[key].V_W = beta * ls[key].V_W + (1-beta) * ls[key].dW
            ls[key].V_b = beta * ls[key].V_b + (1-beta) * ls[key].db
            ls[key].W = ls[key].W*(1-lr*reg/m) - lr * ls[key].V_W
            ls[key].b = ls[key].b*(1-lr*reg/m) - lr * ls[key].V_b
            # ls[key].W = ls[key].W*(1-lr*reg/m) - lr * ls[key].dW
            # ls[key].b = ls[key].b*(1-lr*reg/m) - lr * ls[key].db
        elif(type(ls[key]) == Conv):
            ls[key].V_F = beta * ls[key].V_F + (1-beta) * ls[key].dF
            ls[key].F = ls[key].F*(1-lr*reg/m) - lr * ls[key].V_F
            # print(tc.min(ls[key].dF),tc.max(ls[key].dF))
            # ls[key].F = ls[key].F*(1-lr*reg/m)  - lr * ls[key].dF
        elif(type(ls[key]) == BatchNorm):
            ls[key].gama = ls[key].gama - lr * ls[key].dgama      #*0.01
            ls[key].beta = ls[key].beta - lr * ls[key].dbeta       #*0.01
    return 

from utils.gpu import gpu_acceleration

loss_list = []
acc_list = {"train":[],"valid":[],"test":[]}
def get_accu(Aout,y_labels,data_type,iteration,acc_list):
    A_pred = tc.argmax(Aout,axis = 0)
    acc_num, tot= tc.sum(A_pred.cpu() == y_labels), len(y_labels)
    print(f"{data_type} acc {(acc_num/tot).item()*100:.1f}   {acc_num}/{tot}")
    acc_list[data_type].append([iteration,(acc_num/tot).item()])    
    return
def start_training(train_loader,ls,epochs,lr,reg=3):
    iteration = 0

    for epoch in range(epochs):
        for i, (X,y_labels) in enumerate(train_loader):
            # print("iter", i)
            iteration+=1
            m = X.shape[0]
            Y_hot = tc.nn.functional.one_hot(y_labels,num_classes=10).T #Y_hot: (10,m)
            X,Y_hot = gpu_acceleration(X,Y_hot)
        
            Aout,last_shape = forward_prop(X,ls,train=True)
            backward_prop(Aout,ls,Y_hot,last_shape)
            
            update_params(lr, ls,m,reg)
            if ((epoch % 1) == 0 or (epoch == epochs-1)) and (i%20==0):
                print(f"-----------epoch {epoch}/{epochs}---i {i}/{len(train_loader)}------")
                loss = - tc.sum(Y_hot * tc.log(Aout))/Aout.shape[1]
                # print(f"loss {loss.cpu().item()}")
                loss_list.append([iteration,loss.cpu().item()])

                get_accu(Aout,y_labels,"train",iteration,acc_list)

                X,y_labels = next(iter(val_loader))
                Aout, _ = forward_prop(X.cuda(),ls,train=False)
                get_accu(Aout,y_labels,"valid",iteration,acc_list)

                X,y_labels = next(iter(test_loader))
                Aout, _ = forward_prop(X.cuda(),ls,train=False)
                get_accu(Aout,y_labels,"test",iteration,acc_list)
    return 

if __name__ == "__main__":
    layer_list = {
        'conv0': Conv(f=5,p=2,s=1,n_C_prev=3,n_C_now=32),  #? layer0    (m,3,32,32) --> (m,32,32,32)  
        'bn0'  : BatchNorm(n_C=32),
        'pool1': Pool(f=2,s=2),      # layer1    (m,6,32,32) -->  (m,6,16,16)  
        
        'conv2': Conv(f=5,p=0,s=1,n_C_prev=32,n_C_now=64), #? layer2     (m,32,16,16) -->  (m,64,12,12)            
        'bn2'  : BatchNorm(n_C=64),
        'pool3': Pool(f=2,s=2),   # layer3     (m,16,12,12) --> (m,16,6,6)
        
        'conv4': Conv(f=5,p=0,s=1,n_C_prev=64,n_C_now=128),#layer          (m,64,6,6)  -->  (m,128,2,2)
        'bn4'  : BatchNorm(n_C=128),
        'pool5': Pool(f=2,s=2),# layer5    (m,32,2,2) --> (m,32,1,1)
        
        # 'conv6': Conv(f=5,p=2,s=1,n_C_prev=64,n_C_now=128),#layer          (m,128,4,4)  -->  (m,256,4,4)
        # 'bn6'  : BatchNorm(n_C=128),
        # 'pool7': Pool(f=2,s=2),  # 256,2,2
        
        # 'conv8': Conv(f=5,p=2,s=1,n_C_prev=512,n_C_now=256),#layer          (m,128,4,4)  -->  (m,256,4,4)
        # 'bn8'  : BatchNorm(n_C=256),
        # 'pool9': Pool(f=2,s=2),  # 256,2,2

        'fc0'  : Fc(n_prev=128*1*1,n_now=10),   #* layer6    (32,m)  --> (10,m)  
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


