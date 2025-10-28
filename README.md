# Structured Inference Networks for Nonlinear State Space Models

We aim to model a **sequence of observable events over time**, such as:
- the successive positions of a robot,  
- the heartbeats of a patient,  
- the notes in a piece of music.

Behind these visible observations \( x_t \) lie **internal states \( z_t \)**, which are not directly observable:
- the robot’s true position,  
- the patient’s physiological state,  
- the “harmonic structure” of a musical piece.

These hidden states evolve over time and generate the observations we can see.


## State Space Model

Defined by two equations:

1. **State evolution:**
   \[
   z_t \sim p(z_t \mid z_{t-1})
   \]
   → describes how the world’s state evolves over time.

2. **Observation generation:**
   \[
   x_t \sim p(x_t \mid z_t)
   \]
   → describes how each observation depends on the hidden state.


## The inference problem

We observe a sequence \( x_{1:T} = (x_1, \dots, x_T) \)  
and we aim to **infer the hidden states** \( z_{1:T} = (z_1, \dots, z_T) \).

In theory, we would like to compute:
\[
p(z_{1:T} \mid x_{1:T})
\]
However, this distribution is **analytically intractable** because the relationships are nonlinear and noisy.

We therefore need an **efficient approximation** → this is the role of **Structured Inference Networks (SINs)**.


## Structured Inference Networks (SIN)

A **Structured Inference Network** is a neural network trained to approximate:
\[
q(z_t \mid z_{t-1}, x_{1:T})
\]
In other words: *“What is the latent state at time t, given the previous state and all past and future observations?”*

Unlike a simple model \( q(z_t \mid x_t) \),  
the SIN:
- looks at the **past** to understand system dynamics,  
- looks at the **future** to refine its interpretation of the present.

It is a **bidirectional RNN**, which reads the sequence:
- forward in time (past → future),  
- and backward in time (future → past),  
then combines both representations.


## Deep Markov Model (DMM)

The **Deep Markov Model (DMM)** is a *nonlinear and deep* version of the classic state-space model.

Formally:
\[
\begin{cases}
z_t \sim \mathcal{N}(G_{\alpha}(z_{t-1}), S_{\beta}(z_{t-1})) & \text{(Transition)} \\
x_t \sim p(x_t \mid F_{\kappa}(z_t)) & \text{(Emission)}
\end{cases}
\]

where \( G_\alpha, S_\beta, F_\kappa \) are neural networks (MLPs).  
The DMM learns:
- **how hidden states evolve** (Transition),  
- **how they generate observations** (Emission),  
- **and how to infer them** from data (Combiner).
