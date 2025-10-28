import torch 
import torch.nn as nn
from model.base_modules import Emission, Transition, Combiner, RNN
import data.polyphonic as poly 
from data.seq_util import seq_collate_fn, pack_padded_seq

class DMM(nn.Module):
    def __init__(self, x_dim, z, emission_dim, transition_dim, rnn_dim, rnn_type, rnn_layers, gated, train_init, comb_hidden, use_stlr=False):
        super().__init__()
        self.x_dim = x_dim  #dim des observations
        self.z = z #dimension latente
        self.emission_dim = emission_dim #dim chachées des MLP
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim #taille des états cachés de l'encodeur RNN
        self.rnn_type = rnn_type 
        self.rnn_layers = rnn_layers
        self.gated = gated #transition gated 
        self.train_init = train_init 
        self.comb_hidden = comb_hidden
        self.use_stlr = use_stlr

        #génératif
        self.emitter = Emission(z, emission_dim, x_dim, bernouilli=True)
        self.transition = Transition(z, transition_dim, gated, identity=True)

        #inference
        self.combiner = Combiner(z, rnn_dim, comb_hidden)
        self.encoder = RNN(x_dim, rnn_dim, rnn_layers, dropout=0.0, rnn_type=rnn_type)

        self.mu_p_0, self.logvar_p_0 = self.transition.init_z_0(trainable=train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=train_init)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
    
    def forward(self, x, x_rev, x_seq_lengths):
        device = x.device
        B, T = x.size(0), x.size(1)

        rnn_in = x_rev if self.encoder.reverse_input else x

        pack = pack_padded_seq(rnn_in, x_seq_lengths)
        h_rnn = self.encoder(pack, x_seq_lengths)

        x_reco = torch.zeros(B, T, self.x_dim, device=device)
        mu_q_seq = torch.zeros(B, T, self.z, device=device)
        logvar_q_seq = torch.zeros(B, T, self.z, device=device)
        mu_p_seq = torch.zeros(B, T, self.z, device=device)
        logvar_p_seq = torch.zeros(B, T, self.z, device=device)
        z_q_seq = torch.zeros(B, T, self.z, device=device)
        z_p_seq = torch.zeros(B, T, self.z, device=device)

        mu_p_0 = getattr(self, "mu_p_0", None)
        logvar_p0 = getattr(self, "logvar_p0", None)
        if mu_p_0 is None: 
            mu_p_0 = torch.zeros(self.z, device=device)
            logvar_p0 = torch.zeros(self.z, device=device)
        mu_p_t = mu_p_0.expand(B, self.z)
        logvar_p_t = logvar_p0.expand(B, self.z)

        z_prev = torch.zeros(B, self.z, device=device)

        for t in range(T):
            if t == 0: 
                mu_p, logvar_p = mu_p_t, logvar_p_t
            else:
                mu_p, logvar_p = self.transition(z_prev)
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p

            mu_q, logvar_q = self.combiner(z_prev, h_rnn[:, t, :], h_left=None)
            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q

            z_t = self.reparam(mu_q, logvar_q)
            z_q_seq[:, t, :] = z_t

            z_p = self.reparam(mu_p, logvar_p)
            z_p_seq[:, t, :] = z_p

            x_reco[:, t, :] = self.emitter(z_t)

            z_prev = z_t
        return x_reco, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq
    

    def generate(self, batch_size, seq_len):
        device = self.mu_p_0.device if hasattr(self, "mu_p_0") else "cpu"
        mu_p = getattr(self, "mu_p_0", torch.zeros(self.z, device=device)).expand(batch_size, self.z)
        logv_p = getattr(self, "logvar_p_0", torch.zeros(self.z, device=device)).expand(batch_size, self.z)
        z_prev = None 

        z_p_seq      = torch.zeros(batch_size, seq_len, self.z, device=device)
        mu_p_seq     = torch.zeros(batch_size, seq_len, self.z, device=device)
        logvar_p_seq = torch.zeros(batch_size, seq_len, self.z, device=device)
        x_logits     = torch.zeros(batch_size, seq_len, self.x_dim, device=device)

        for t in range(seq_len):
            mu_p_seq[:, t, :]     = mu_p
            logvar_p_seq[:, t, :] = logv_p

            z_t = self.reparam(mu_p, logv_p)
            z_p_seq[:, t, :] = z_t

            x_logits[:, t, :] = self.emitter(z_t)

            mu_p, logv_p = self.transition(z_t)

        return x_logits, z_p_seq, mu_p_seq, logvar_p_seq