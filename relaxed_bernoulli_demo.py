import torch
from torch import nn, optim
'''
This code optimizes the parameter of a Bernoulli distribution so that 
samples from that distribution are tuned to satisfy some loss function. 
This is done by using the Relaxed Bernoulli.

The Relaxed Bernoulli is a distribution over (0,1) that continuously 
approximates a Bernoulli. The approximation is determined by the 
temperature parameter. If the temperature is close to 0, the Relaxed 
Bernoulli becomes discrete with distribution determined by logits/probs. 
If the temperature goes to inf, the Relaxed Bernoulli puts all mass on the 
value 0.5. 

The samples from the Relaxed Bernoulliare reparameterize-able, which means that 
a sample from the Relaxed Bernoulli can be written as a differentiable transformation 
of a draw from a base distribution, where the draw from the base distribution 
doesn't depend on the parameters we want to optimize. It was discovered concurrently by:
    
    [1] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. 
        The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. 
        2016.

    [2] Eric Jang, Shixiang Gu, and Ben Poole. 
        Categorical Reparameterization with Gumbel-Softmax. 2016.

To actually use the samples from a Relaxed Bernoulli as Bernoulli samples, we
round the value in (0,1) to 0 or 1, and copy the gradient over the rounding step.
To do this, we define a rounding function that uses torch.round in the forward pass
and copies the gradient in the backward pass. Then we define a Relaxed Bernoulli that 
applies our rounding function in the rsample (reparameterized sample) method.
'''
class MyRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx,grad):
        return grad.clone()

class MyRelaxedBernoulli(torch.distributions.RelaxedBernoulli):
    def rsample(self,*args,**kwargs):
        sample = super(MyRelaxedBernoulli,self).rsample(*args,**kwargs)
        return MyRound.apply(sample)

'''
let us optimize an unconstrained variable called _prob, 
and use the variable nn.Sigmoid()(_prob).
'''
temp = torch.tensor([2.2])
_prob = torch.tensor([-1.0])
_prob.requires_grad=True
optimizer = optim.Adam([_prob],lr=0.001)

'''
For this demo, let's initialize _prob near 0 so that nn.Sigmoid()(_prob) is close to 0.5.
Then, let's use a loss function that says "I want '1' samples from the Bernoulli". We expect,
through grad descent, that _prob should become very positive so that nn.Sigmoid()(_prob) approaches 1
'''
def loss_function(x):
    return (1-x).pow(2).sum()

for i in range(10000):
    prob = nn.Sigmoid()(_prob)
    p = MyRelaxedBernoulli(temp,probs=prob)
    x = p.rsample()
    if i%100==0:
        print('---')
        print("_prob is:",_prob)
        print("prob is:",prob)
        print("sample is:",x)
    loss = (1-x).pow(2).sum()
    loss.backward()
    optimizer.step()
