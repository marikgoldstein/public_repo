import torch
from torch import nn, optim

'''
See notes in relaxed_bernoulli_demo.py in this same repo for
an explanation on relaxed bernoulli. Relaxed categorical follows by analogy.
These distributions were introduced concurrently by:

    [1] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. 
        The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. 
        2016.

    [2] Eric Jang, Shixiang Gu, and Ben Poole. 
        Categorical Reparameterization with Gumbel-Softmax. 2016.

In the case of the categorical, we must use argmax rather than the rounding
we did for the relaxed bernoulli.
'''
class MyArgmax(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input)
        ret = torch.zeros_like(input)
        ret[idx] += 1.0
        return ret        

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()

class MyRelaxedCat(torch.distributions.RelaxedOneHotCategorical):
    def rsample(self,*args,**kwargs):
        sample = super(MyRelaxedCat,self).rsample(*args,**kwargs)
        return MyArgmax.apply(sample)
'''
let us optimize an unconstrained variable called _probs, 
and use the variable nn.Softmax()(_probs).
'''
temp = torch.tensor([2.2])
_probs = torch.randn(6)
_probs.requires_grad=True
optimizer = optim.Adam([_probs],lr=0.001)

'''
For this demo, let's use a loss function that says "I want '5' samples from the categorical
defined over (0,1,2,3,4,5).
'''
def loss_function(x):
    want = torch.tensor([0.,0.,0.,0.,0.,1.])
    return (want-x).pow(2).sum()

for i in range(10000):
    probs = nn.Softmax()(_probs)
    p = MyRelaxedCat(temp,probs=probs)
    x = p.rsample()
    if i%100==0:
        print('---')
        print("_probs is:",_probs)
        print("probs is:",probs)
        print("sample is:",x)
    loss = loss_function(x)
    loss.backward()
    optimizer.step()
