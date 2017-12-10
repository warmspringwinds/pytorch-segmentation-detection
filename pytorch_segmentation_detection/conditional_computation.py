import torch
import torch.nn as nn
from torch.autograd import Variable


def sample_gumbel(shape, eps=1e-10):
    """Returns samples from Gumble distribution with parameters (0, 1) of
    a specified shape.
    
    Based on https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    and https://github.com/pytorch/pytorch/pull/3341. (@hughperkins)
    
    Parameters
    ----------
    shape : tuple of ints
        Output shape of samples
    eps : float
        Constant that is added for numerical stability
        
    Returns
    -------
    gumble_samples_tensor : Tensor
        Tensor that contains random samples
    """
    
    uniform_samples_tensor = torch.rand(shape)
    gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
    
    return gumble_samples_tensor


def sample_gumbel_like(template_tensor, eps=1e-10):
    """Returns samples from Gumble distribution with parameters (0, 1) of
    a type and shape as template_tensor.
    
    Based on https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    and https://github.com/pytorch/pytorch/pull/3341. (@hughperkins)
    
    Parameters
    ----------
    template_tensor : Tensor
        Tensor which shape and type will be used while creating a gumble_samples_tensor
    eps : float
        Constant that is added for numerical stability
        
    Returns
    -------
    gumble_samples_tensor : Tensor
        Tensor that contains random samples
    """
    
    uniform_samples_tensor = template_tensor.clone().uniform_()
    gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
    
    return gumble_samples_tensor


def gumbel_softmax_sample(logits, tau=1, dim=None):
    """Returns differentiable samples from specified log-probabilities (logits variable).
    
    Argmax in the Gumble sampling reparametrization trick is approximated with softmax.
    See more for details:
    # https://arxiv.org/abs/1611.01144
    
    Based on https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    and https://github.com/pytorch/pytorch/pull/3341. (@hughperkins)
    
    Parameters
    ----------
    logits : Variable
        Variable with log-probabilities of a discrete distribution
    tau : float
        The discrete distribution approximation constant (See above-mentioned paper for more details)
    dim : int
        Dimension of the Varialbe along which the sampling should be performed
        
    Returns
    -------
    soft_samples : Variable
        Variable with soft samples
    """
    
    dim = logits.size(-1) if dim is None else dim
                
    gumble_samples_tensor = sample_gumbel_like(logits.data)
    
    # Next line is equivalent to sampling from discrete distribution
    # if we apply argmax. Here, softmax is used instead to make it
    # differentiable. But still arxmax can be used during forward inference
    # while approximating the derivative with softmax during the backward pass.
    # See here for more details:
    # Here, it is clearn why ppl call it log probabilities also
    # https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    # Original paper
    # https://arxiv.org/abs/1611.01144
    
    gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
    
    soft_samples = nn.functional.softmax(gumble_trick_log_prob_samples / tau, dim)
    
    return soft_samples


def gumbel_softmax(logits, dim=None, hard=False, tau=1):
    """Returns differentiable samples from specified log-probabilities (logits variable) of discrete distribution.
    Has the same API as torch.nn.functional.softmax().
    
    Argmax in the Gumble sampling reparametrization trick is approximated with softmax.
    See more for details:
    # https://arxiv.org/abs/1611.01144
    
    Can also perform the hard sampling and approximate gradient with the gumble-softmax distribution.
    
    Based on https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    and https://github.com/pytorch/pytorch/pull/3341. (@hughperkins)
    
    Parameters
    ----------
    logits : Variable
        Variable with log-probabilities of a discrete distribution
    dim : int
        Dimension of the Varialbe along which the sampling should be performed
    hard : boolean
        Whether or not to perform hard sampling or soft (See Section 2. of the paper)
    tau : float
        The discrete distribution approximation constant (See above-mentioned paper for more details)
        
    Returns
    -------
    return_samples : Variable
        Variable with soft or hard differentiable samples
    """
    
    dim = logits.size(-1) if dim is None else dim
    
    samples_soft = gumbel_softmax_sample(logits, tau=tau, dim=dim)
        
    if hard:
        
        _, max_value_indexes = samples_soft.data.max(dim, keepdim=True)
        
        samples_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1.0)
        
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        return_samples = Variable(samples_hard - samples_soft.data) + samples_soft
        
    else:
        return_samples = samples_soft
        
    return return_samples