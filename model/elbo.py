import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""
Objectif : maximiser la borne inférieure ELBO
    
-> Combiner négative log-vraisemblance (NLL) & KL Divergence
"""

def kl_div(mu_q, mu_p, logvar_q, logvar_p):
    """
    KL Divergence entre deux gaussienne q, p avec des moyennes et covariances respectives mu_q, sum_q, mu_p, sum_p : 

    KL(q || p) = (1/2) * [(logvar_p / logvar_q) + (logvar_q + (mu_q - mu_p)**2) / logvar_p - 1]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    return 0.5 * (
        logvar_p - logvar_q + 
        (var_q + (mu_q - mu_p)**2) / var_p - 1
    )


def nll(x_hat, x, logits=True):
    """
    Negative log-likelihood 
    """
    if logits: 
        loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='none')
    else:
        loss = F.binary_cross_entropy(x_hat, x, reduction='none')
    return loss


def loss(x_hat, x, mu_q, mu_p, logvar_q, logvar_p, annealing_factor=1.0, logits=True):
    """
    Loss = NLL + KL

    - Moyenne ou somme sur les dimensions 
    - Annealing sur la KL Divergence 
    - S'assurer que x_hat et x sont de même nature 
    """
    kl_r = kl_div(mu_q, mu_p, logvar_q, logvar_p)
    nll_r = nll(x_hat, x, logits=True)

    kl_time = kl_r.sum(dim=-1)
    nll_time = nll_r.sum(dim=-1)

    kl_mean = kl_time.mean()
    nll_mean = nll_time.mean()

    loss = kl_mean * annealing_factor + nll_mean

    return {
        'kl_raw' : kl_r, 
        'nll_raw' : nll_r, 
        'kl_time' : kl_time, 
        'nll_time' : nll_time, 
        'kl_mean' : kl_mean, 
        'nll_mean' : nll_mean, 
        'loss' : loss
    }