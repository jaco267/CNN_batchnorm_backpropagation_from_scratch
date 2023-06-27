import torch as tc
def gpu_acceleration(*args):
    if( not tc.cuda.is_available()):
        return args

    args = list(args)  #tuple --> args
    for i in range(len(args)):
        args[i] = args[i].cuda()
    return args