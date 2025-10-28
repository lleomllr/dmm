import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Transition(nn.Module):
    """
    Gated Transition Function 
    
    parameterization :
        g_t = MLP(z_{t-1}, ReLU, sigmoid)
        h_t = MLP(z_{t-1}, ReLU, Identity function)

        mu_t(z_{t-1}) = (1 - g_t) x (W_{mu_p} z_{t-1} + b_{mu_p}) + g_t x h_t 
        sig_t^2(z_{t-1}) = softplus(W_{sig_p^2} ReLU(h_t) + b_{sig_p^2})
    """

    def __init__(self, z, transition, gated=True, identity=True):
        super().__init__()
        self.z = z
        self.transition = transition
        self.gated = gated
        self.identity = identity

        self.lin1p = nn.Linear(z, transition)
        self.lin2p = nn.Linear(transition, z)

        if gated:
            #MLP qui produit g_t
            self.lin1 = nn.Linear(z, transition)
            self.lin2 = nn.Linear(transition, z)
            #transfo linéaire W_{mu p}z_t-1
            self.lin_n = nn.Linear(z, z)

        self.lin_v = nn.Linear(transition, z)

        if gated and identity:
            #init partie linéaire W_{mu p} proche de l'identité pour stabiliser les transitions au début du training 
            self.lin_n.weight.data = torch.eye(z)
            self.lin_n.bias.data = torch.zeros(z)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z_t_1):
        #h_t = ReLU(W_h z_t-1 + b_h)
        h_t = self.relu(self.lin1p(z_t_1))
        #mu_r = W_muh h_t + b_muh
        mu_r = self.lin2p(h_t)
        #sig^2_t = softplus(W_sig^2 h_t + b_sig^2) - softplus : variances restent > 0
        sig_t = F.softplus(self.lin_v(h_t)) + 1e-8

        if self.gated:
            g_h = self.relu(self.lin1(z_t_1))
            g_t = self.sigmoid(self.lin2(g_h))
            #mu_t = (1 - g_t) . (W_mup z_t-1) + g_t . h_t
            mu = (1 - g_t) * self.lin_n(z_t_1) + g_t * mu_r
        else:
            mu = mu_r

        logvar = torch.log(sig_t)
        return mu, logvar
    

class Emission(nn.Module):
    """
    Emission Function 

    parameterize the emission function F_k with 2-layer MLP : 
        MLP(x, NL_1, NL_2) = NL_2(W_2 NL_1(W_1 x + b_1) + b_2), where NL : ReLU, sigmoid, tanh

        F_k(z_t) = sigmoid(W_emission MLP(z_t, ReLU, ReLU) + b_emission) : mean probas of independent Bernoullis

        p_theta(x_t | z_t) = Bernouilli(mu_x(z_t))
    """

    def __init__(self, z, emission, x_dim, bernouilli=True):
        super().__init__()
        self.z = z
        self.emission = emission
        self.x_dim = x_dim 
        self.bernouilli = bernouilli

        self.lin1 = nn.Linear(z, emission)
        self.lin2 = nn.Linear(emission, emission)
        self.lin3 = nn.Linear(emission, x_dim)

        if not bernouilli:
            self.lin_var = nn.Linear(emission, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.relu(self.lin1(z_t))
        h2 = self.relu(self.lin2(h1))
        out = self.lin3(h2)
        return out


class Combiner(nn.Module):
    """
    Combiner Function 

    parameterize : 
        variational distribution q(z_t | z_t-1, x_{1:T}) : diagonal Gaussian distrib 

    ST-LR : h = 1/3(tanh(W z_t-1 + b) + h_left + h_right)
    DKS : h = 1/2(tanh(W z_t-1 + b) + h_right)

    h = tanh(W z_t-1 + b) + h_t
    mu_t = W_mu h + b_mu 
    sig^2_t = softplus(W_sig^2 h + b_sig^2)
    """

    def __init__(self, z, h, hidden_state):
        super().__init__()
        self.z = z
        self.h = h
        self.hidden = hidden_state

        self.lin1 = nn.Linear(z + h, hidden_state)
        self.lin2 = nn.Linear(hidden_state, z)
        self.lin3 = nn.Linear(hidden_state, z)

    def init_z(self, trainable=True):
        mu = nn.Parameter(torch.zeros(self.z), requires_grad=trainable)
        logvar = nn.Parameter(torch.zeros(self.z), requires_grad=trainable)
        return mu, logvar
    
    def forward(self, z_t_1, h_right, h_left):
        if h_left is None:
            #DKS
            h_comb = (1/2) * (torch.tanh(self.lin1(torch.cat([z_t_1, h_right], dim=-1))) + h_right) 
        else:
            #ST-LR
            h_comb = (1/3) * (torch.tanh(self.lin1(torch.cat([z_t_1, h_left, h_right], dim=-1))) + h_left + h_right)

        mu_t = self.lin2(h_comb)
        sig_t = F.softplus(self.lin3(h_comb))
        logvar_t = torch.log(sig_t)

        return mu_t, logvar_t
    

class RNN(nn.Module):
    """
    Lit la séquence à l'envers pour approximer h_t en utilisant x_{t:T}

    """
    def __init__(self, x_dim, rnn_dim, n_layer=1, dropout=0.0, rnn_type='rnn'):
        super().__init__()
        self.x_dim = x_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.dropout = dropout
        
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=x_dim, hidden_size=rnn_dim, nonlinearity='relu', batch_first=True, num_layers=n_layer, dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=x_dim, hidden_size=rnn_dim, batch_first=True, num_layers=n_layer, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=x_dim, hidden_size=rnn_dim, batch_first=True, num_layers=n_layer, dropout=dropout)

    def forward(self, x):
        x_rev = torch.flip(x, dims=[1])
        h_rev, _ = self.rnn(x_rev)
        h_right = torch.flip(h_rev, dims=[1])
        return h_right