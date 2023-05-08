import numpy as np
import math
import matplotlib.pyplot as plt 
import torch

def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w

class CPG:
    def __init__(self, MI=0.0):
        self.MI = MI
        self.o1 = 0.01
        self.o2 = 0.01
        self.w11 = 1.4
        self.w22 = 1.4
        self.w12 = -(self.MI+0.18)
        self.w21 =  (self.MI+0.18)

    def step(self):
        self.o1 = math.tanh( self.o1*self.w11 + self.o2*self.w12)
        self.o2 = math.tanh( self.o2*self.w22 + self.o1*self.w21)

    def get_weights(self):
        return [[self.w11,self.w12],
                [self.w21,self.w22]]
    
    def get_output(self):
        return [self.o1, self.o2]


class RBFNet:
    def __init__(self, input_size, output_size, kernel_num = 10):
        """
        input_size = number of inputs to the RBF network
        kernel_num = 10: number of rbf kernel
        """
        self.weights = torch.Tensor(kernel_num, output_size).uniform_(-1,1)
        self.center = torch.Tensor((input_size, kernel_num)).normal_(0,1)
        self.sigma = torch.Tensor((input_size, kernel_num)).zero_()


    def forward(self, pre):

        with torch.no_grad():
            pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            
            for i, W in enumerate(self.weights):
                # post = torch.tanh(pre @ W.float())
                post = torch.tanh(pre @ W.double())

                pre = post

        return post.detach().numpy()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )
        return p.flatten().numpy()


    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        for i, w in enumerate(self.weights):
            a, b = w.shape
            self.weights[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 


    def get_weights(self):
        return [w for w in self.weights]

if __name__ == "__main__":
    cpg = CPG(MI=0.05)
    cpg_outputs = np.array([[0,0]])
    # counter
    for i in range(500):
        cpg_outputs = np.append(cpg_outputs, [cpg.get_output()], axis=0)
        cpg.step()
    print(cpg_outputs)
    plt.plot(cpg_outputs[:,0])
    plt.plot(cpg_outputs[:,1])
    plt.show()