import numpy as np
from torch.autograd import Variable
import torch as torch 
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image,net,num_classes=10,overshoot=0.02,max_iter=50):


    is_cuda=torch.cuda.is_available()
    if is_cuda:
        print("Using GPU")
        image=image.cuda
        net=net.cuda()

    else:
        print("UsingCPU")

    f_image=net.forward(Variable(image[None,:,:,:],requires_grad=True)).data.cpu().numpy.flatten()
    I=(np.array(f_image)).flatten().argsort()[::-1]
    I=I[0:num_classes]
    label=I[0]
    
    input_shape=image.cpu().numpy().shape
    pert_image=copy.deepcopy(image)
    w=np.zeros(input_shape)
    r_tot=np.zeros(input_shape)

    loop_i=0

    x=Variable(pert_image[None,:],requires_grad=True)
    fs=net.forward(x)
    fs_list=[fs[0,I[k]] for k in range(num_classes)]
    k_i=label

    while k_i ==label and loop_i < max_iter:
        pert=np.inf
        fs[0,I[0]].backward(retain_graph=True)
        grad_orig=x.grad.data.cpu().numpy().copy()

        for k in range(1,num_classes):
            zero_gradients(x)
            fs[0,I[k]].backward(retain_graph=True)
            cur_grad=x.grad.data.cpu().numpy().copy()
            
                   