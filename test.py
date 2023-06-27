#%%
from _0_preprocessing import test_loader
import torch as tc
import pickle
from train4 import forward_prop,Fc, Conv,Pool,BatchNorm,Relu


outdir = './_train_ML_data/train_W_b_result'
with open(f"{outdir}/v2_modulization.pkl",'rb') as cnnInfoPickle:
    cnnInfo = pickle.load(cnnInfoPickle)
ls = cnnInfo['layerList']

X_test,y_test = iter(test_loader).next()
X_test,y_test = X_test.cuda(),y_test.cuda()

Aout,_  = forward_prop(X_test,ls)
A_pred = tc.argmax(Aout,axis = 0)

currect_num = tc.sum(A_pred == y_test)
tot = len(y_test)
accu = currect_num/tot
print(f"{accu}   {currect_num}/{tot}   ")

acc_list,loss_list = cnnInfo["acc_list"],cnnInfo["loss_list"]
import numpy as np
import matplotlib.pyplot as plt
def plot_learning_curve(learning_curve_list,ax):
    x_axis,y_axis = [],[]
    for item in learning_curve_list:
        x_axis.append(item[0])
        y_axis.append(item[1])
    
    X, Y=np.array(x_axis),np.array(y_axis)
    ax.plot(X,Y)
    ax.set_title('Learning curve',fontname="Arial",fontsize=20,\
        weight='bold',style='italic')
    ax.set_xlabel("iterations",fontsize=20)
    ax.set_ylabel("Cross entropy loss",fontsize=20)
    ax.tick_params(axis='y', labelrotation=0)
    return 
def plot_train_accu(acc_list,ax):
    for key in acc_list:
        # print(key)
        x_axis,y_axis = [],[]
        for item in acc_list[key]:
            x_axis.append(item[0])
            y_axis.append(item[1])
        
        X, Y=np.array(x_axis),np.array(y_axis)
        
        ax.plot(X,Y,label = key)


    ax.set_title('Training accuracy',fontname="Arial",fontsize=20,\
        weight='bold',style='italic')
    ax.set_xlabel("iterations",fontsize=20)
    ax.set_ylabel("Accuracy rate",fontsize=20)
    ax.tick_params(axis='y', labelrotation=0)
    ax.legend()
    return 


conv_list,nn_list = [], []
for key in ls:
    if type(ls[key]) == Fc:
        # print("fc",ls[key].W.shape)
        nn_list.append(ls[key].W.cpu().reshape(-1).tolist())
    elif type(ls[key]) == Conv:
        # print("conv",ls[key].F.shape)
        conv_list.append(ls[key].F.cpu().reshape(-1).tolist())

fig_x,fig_y = 4,2
fig,ax=plt.subplots(fig_x,fig_y,figsize=(16,10))
for i in range(fig_x):
    for j in range(fig_y):
        ax[i][j].set_xlabel("values",fontname="Arial",fontsize=15)
        ax[i][j].set_ylabel("numbers",fontname="Arial",fontsize=15)
h_range = [-0.8,0.8] 
plot_train_accu(acc_list,ax[0][0])
plot_learning_curve(loss_list,ax[0][1])


ax[1][0].set_title('conv0',fontname="Arial",fontsize=20)
ax[1][0].hist(conv_list[0],range=h_range,bins=50)
ax[2][0].set_title('conv1',fontname="Arial",fontsize=20)
ax[2][0].hist(conv_list[1],range=h_range,bins=50)
ax[3][0].set_title('conv2',fontname="Arial",fontsize=20)
ax[3][0].hist(conv_list[2],range=h_range,bins=50)

h_range = [-0.8,0.8] 
ax[1][1].set_title('dense0',fontname="Arial",fontsize=20)
ax[1][1].hist(nn_list[0],range=h_range,bins=150)

plt.tight_layout()
plt.show()





print()
"""
#*    1-2
index_wrong = tc.where((A_pred!=y_test) & (y_test==4))[0].tolist()
# index = tc.where((A_pred!=y_test))[0].tolist()
# print(f"wrong indices {len(index_wrong)}\n",index_wrong)

img_num = 4
print("predict value\n",A_pred[index_wrong[0:img_num]])
print("label value\n",y_test[index_wrong[0:img_num]])
X_test = X_test.cpu()  #10000,1,28,28
img_wrong = X_test[index_wrong[0:img_num]] #6,1,28,28 



print("-------------wrong predictions----------")
for i in range(img_num):
    fig = plt.subplot(2,2,i+1)
    fig.set_xlabel(f"pred: {A_pred[index_wrong[i]]}, label: {y_test[index_wrong[i]]}",fontsize=13)
    plt.tight_layout()
    plt.imshow(img_wrong[i][0],cmap='gray')
plt.show()

index_correct = tc.where(y_test == 4)[0].tolist()
img_correct = X_test[index_correct[0:img_num]]

for i in range(img_num):
    fig = plt.subplot(2,2,i+1)
    fig.set_xlabel(f"pred: {A_pred[index_correct[i]]}, label: {y_test[index_correct[i]]}",fontsize=13)
    plt.tight_layout()
    plt.imshow(img_correct[i][0],cmap='gray')
plt.show()

fig = plt.subplot(1,2,1)
plt.imshow(img_wrong[2][0],cmap='gray')
fig.set_xlabel(f"pred: {A_pred[index_wrong[2]]}, label: {y_test[index_wrong[2]]}",fontsize=13)

fig = plt.subplot(1,2,2)
plt.imshow(img_correct[2][0],cmap='gray')
fig.set_xlabel(f"pred: {A_pred[index_correct[2]]}, label: {y_test[index_correct[2]]}",fontsize=13)

plt.tight_layout()
plt.show()


def plot_outputs(inputX,layerList):
    layerList, Aout  = forward_prop(inputX,layerList)

    conv_layer0 = layerList[0]
    for key in conv_layer0.keys():
        print(key,end="  ")
    print()
    output0 = conv_layer0['Z']
    # print(output0.shape)  # 1,6,28,28
    conv0_filter_num = output0.shape[1]
    for i in range(conv0_filter_num):
        fig = plt.subplot(2,3,i+1)
        plt.imshow(output0[0][i].cpu(),cmap='gray')
    plt.show()

    conv_layer1 = layerList[2]
    output1 = conv_layer1['Z']
    # print(output1.shape) #1,16,10,10
    conv1_filter_num = output1.shape[1]

    fig, axs = plt.subplots(2,1,figsize=(16,16))
    for i in range(conv1_filter_num):
        fig = plt.subplot(4,4,i+1)
        plt.imshow(output1[0][i].cpu(),cmap='gray')
    plt.show()

i_shape = img_wrong[2].shape
img_wrong_test = img_wrong[2].reshape(1,i_shape[0],i_shape[1],i_shape[2]).cuda()
plot_outputs(img_wrong_test,layerList)
img_correct_test = img_correct[2].reshape(1,i_shape[0],i_shape[1],i_shape[2]).cuda()
plot_outputs(img_correct_test,layerList)
print()
"""

print()