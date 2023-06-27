# %%
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import torch as tc

#len(mnist_testset)   10000
def loadCIFAR_10(batch_size,trans=True):
    # dataset has PILImage images of range [0, 1]. 
    # We transform them to Tensors of normalized range [-1, 1]
    global train_dataset, test_dataset
    if trans:
        # channel=（channel-mean）/std       (0~1)-0.5 = (-0.5 ~ -0.5) --> (-0.5 ~ -0.5)/0.5 = (-1,1)  
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5),std= (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    data_dir_path = "./_python_ML_data"
    # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=True,download=True, transform=transform)
    
    train_subset, val_subset = tc.utils.data.random_split(train_dataset, [45000, 5000])
    
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir_path, train=False,download=True, transform=transform)
         
    train_loader = tc.utils.data.DataLoader(train_subset, batch_size=batch_size,shuffle=True)
    val_loader = tc.utils.data.DataLoader(val_subset, batch_size=1000000,shuffle=False)
    test_loader = tc.utils.data.DataLoader(test_dataset, batch_size=1000000, shuffle=False)

    return train_loader,val_loader, test_loader

batch_size = 128
train_loader,_, __ = loadCIFAR_10(batch_size,trans=False)
X_train_raw, y_train_raw = next(iter(train_loader))  #128,3,32,32  tc.float32  #128  tc.int64
print(tc.max(X_train_raw),tc.min(X_train_raw),tc.mean(X_train_raw),tc.var(X_train_raw))  #*  1 0


train_loader,val_loader, test_loader = loadCIFAR_10(batch_size,trans=True)

X_train, y_train = next(iter(train_loader))  #128,3,32,32  tc.float32  #128  tc.int64
print(tc.max(X_train),tc.min(X_train),tc.mean(X_train),tc.var(X_train))         #*  1 -1
print(X_train.shape)


X_val,y_labels = next(iter(val_loader))
X_test,y_labels = next(iter(test_loader))
# %%

