# Deep Markov Model (DMM)

PyTorch re-implementation of **Structured Inference Networks for Nonlinear State Space Models** (Krishnan, Shalit & Sontag, AAAI 2017) [pdf](https://arxiv.org/pdf/1609.09869)

---

## Theoric Overview

Aim to model a **sequence of observable events over time**, such as:
- the successive positions of a robot,  
- the heartbeats of a patient,  
- the notes in a piece of music.

Behind these visible observations $\( x_t \)$ lie **internal states $\( z_t \)$**, which are not directly observable:
- the robot’s true position,  
- the patient’s physiological state,  
- the “harmonic structure” of a musical piece.

These hidden states evolve over time and generate the observations we can see.

---

### State Space Model (SSM)

Defined by two key distributions:


$z_t &\sim p(z_t \mid z_{t-1}) \quad &\text{(state transition)}$ \\
$x_t &\sim p(x_t \mid z_t) \quad &\text{(observation emission)}$


We only observe $\( x_{1:T} \)$, but we want to infer the hidden sequence $\( z_{1:T} \)$.  
Exact inference is intractable — hence the need for a **Structured Inference Network**.

---

### Structured Inference Network (SIN)

Instead of approximating each latent variable independently as $\( q(z_t \mid x_t) \)$,  
the SIN learns a **structured posterior** that accounts for temporal dependencies:


$q(z_t \mid z_{t-1}, x_{1:T})$


It uses a **bidirectional RNN** that reads the sequence both forward and backward in time  
to leverage **past and future context** for better inference.

---

### Deep Markov Model (DMM)

The Deep Markov Model is a nonlinear, neural version of the state-space model:


$z_t \sim \mathcal{N}(G_\alpha(z_{t-1}), S_\beta(z_{t-1})) & \text{Transition network}$ \\
$x_t \sim p(x_t \mid F_\kappa(z_t)) & \text{Emission network}$


It consists of:
- a **Transition** network (evolves hidden state),
- an **Emission** network (generates observations),
- a **Combiner** network (approximates posterior),
- and a **RNN encoder** (encodes the sequence structure).

The training objective maximizes the **Evidence Lower Bound (ELBO):**


$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) \| p_\theta(z))$

---

## Implementation 

**Main modules:**
- `Transition`: gated transition dynamics $\( p(z_t | z_{t-1}) \)$  
- `Emission`: reconstructs $\( x_t \)$ from $\( z_t \)$  
- `Combiner`: approximates $\( q(z_t | z_{t-1}, x_{1:T}) \)$  
- `RNN`: encoder producing hidden summaries of observations  

**Training objective:**  

$\mathcal{L} = \text{NLL} + \beta \cdot KL$

where NLL = negative log-likelihood, KL = regularization,  
and β is an annealing factor for stability.

---

## Dataset: Nottingham Polyphonic Music

**Nottingham** dataset of polyphonic MIDI sequences.

### Data preparation
Convert `.mat` files to `.pkl` once:
```bash
python convert.py --input ./data/nottingham.mat --output ./data/nottingham.pkl
