### Expected Maximization algorithm and Baum-Welch algorithm

When reading about arbitrary optimalization and variations of the maximum-likelyhood method I stumbled accross the EM and Baum Welch algorithms, and I thought I'd experiment with them and see how they work.

In this repository I have made three small applications using these, from a very simple dice bias-parameter system to an investment portfolio implementing the Baum Welch algorithm to try to estimate market regime parameters.

### The Baum-Welch algorithm for the weather-system HMM

Given a hidden Markov-chain (HMM), consisting of a set of states (which are hidden) and observations, the goal is to estimate the parameters of the system. Here, the parameter set consists of the transition matrix $A$, the emission matrix $B$ and the initial starting probabilities $\pi$. If $N$ is the amount of states, $M$ the amount of possible observations in each timestamp and $T$ the amount of timestamps of the system (discrete integers), then $A$ is a $N\times N$ matrix where $a_{i,j}=\mathbb{P}(S_{t+1}=j\ |\ S_t=i\text{ for all }t)$ (being the probability that the next state $S_{t+1}$ will be $j$ given the current state $i$), $B$ is a $N\times M$ matrix where $b_{i,k}=b_i(O_t)=\mathbb{P}(O_t=k\ |\ S_t=i\text{ for all }t)$ (being the probability that we observe $O_t$ to be k given that we are in state $i$) and $\pi$ is a vector of length $N$ where $\pi_i=\mathbb{P}(S_1=i)$ (being the probability that the HMM started in state $i$).

The first step in the algorithm is making a guess for the values of $A$, $B$ and $\pi$ to initialize them.

The second step is called the expectation-step, where we estimate the probability that the observations that we saw came from the parameters $\theta=(A,B,\pi)$ that we currently have.

This step consists of the so-called forward and backward algorithm.  
In the forward algorithm, we estimate the probability that the observed sequence of the HMM from $t=1$ to $t=t$, ended in state $i$, we denote this by $\alpha_t(i)=\mathbb{P}(O_1,...,O_t,S_t=i)$. For this, we first initialize $\alpha_1(i)=\pi_i b_i(O_1)$, (the probability that we were in state $i$ at the beginning times the probability that the first observation ($O_1$) occured during state $i$). Then, we compute  
$\alpha_t(j)=b_j(O_t)\sum_i\alpha_{t-1}(i)a_{i,j}$  
recursively. So we consider all ways to reach state $j$ from previous states $\alpha_{t-1}(i)a_{i,j}$ multiplied by the probability of seeing observation $O_t$ in state $j$.

In the backward algorithm, we do the opposite: we estimate the probability of observing the future from $t=t+1$ onward, given state $i$ at time $t$. We denote this by $\beta_t(i)=\mathbb{P}(O_{t+1},...,O_T\ |\ S_t=i)$. We initialize $\beta_T(i)=1$ (since there is no future at $t=T$), and then we compute recursively again:  
$\beta_t(i)=\sum_j a_{i,j}b_j(O_{t+1})\beta_{t+1}(j)$  
So here we are considering the sum over all possible future states $j$ (the sum of the product of the probability of moving from state $i$ to $j$ times the probability of observing $O_{t+1}$ during state $j$ times the probability of having gotten to the next state $S_{t+1}$).

Next, we compute $\gamma\in\mathbb{R}^{T\times N}$ where $\gamma_t(i)=\mathbb{P}(S_t=i\ |\ O_1,...,O_T)$ (the probability that we were in state $i$ at time $t$ given the observations), by  
$$\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(j)\beta_t(j)}$$  
The way we get this formula, is by transforming the formula for gamma: $\gamma_t(i)=\mathbb{P}(S_t=i\ |\ O_1,...,O_T)=\mathbb{P}(S_t=i,O_1,...,O_T)/\mathbb{P}(O_1,...,O_T)$ and splitting $O_1,...,O_T$ into $O_1,...,O_t$ and $O_{t+1},...,O_T$ for each $t$, and because $\mathbb{P}(S_t=i,O_1,...,O_t)=\alpha_t(i)$ and $\mathbb{P}(S_t=i,O_{t+1},...,O_T)=\beta_t(i)$ we get the formula for the numerator. The formula for the denominator follows from the fact that $\mathbb{P}(O_1,...,O_T)$ is just the sum over all states $j$ of $\alpha_t(j)\beta_t(j)$ (we are considering the probability of having seen the given observations, no matter the end state).

Then we compute $\xi\in\mathbb{R}^{T\times N\times N}$ where $\xi_t(i,j)=\mathbb{P}(S_t=i,S_{t+1}=j\ |\ O_1,...,O_T)$ (the probability that we were in state $i$ at time $t$ and that the next state $S_{t+1}$ will be $j$, given the observations), by  
$$\xi_t(i,j)=\frac{\alpha_t(i)a_{i,j}b_j(O_{t+1})\beta_{t+1}(j)}{\sum_{k,l}\alpha_t(k)a_{k,l}b_l(O_{t+1})\beta_{t+1}(l)}$$  
We obtain this equation by further developing the formula for xi: $\xi_t(i,j)=\mathbb{P}(S_t=i,S_{t+1}=j\ |\ O_1,...,O_T)=\mathbb{P}(S_t=i,S_{t+1}=j,O_1,...,O_T)/\mathbb{P}(O_1,...,O_T)$  
In the numerator, $\mathbb{P}(S_t=i,S_{t+1}=j,O_1,...,O_T)=\alpha_t(i)a_{i,j}b_j(O_{t+1})\beta_{t+1}(j)$, we count the probability of having gotten to state $i$ at time $t$ whilst having transitioned into the next state $j$, whilst also having observed state $j$ at time $t+1$, multiplied by the probability of observing $O_{t+1},...,O_T$ in the future (being $\beta_{t+1}(j)$). In the denominator, we simply compute the same but summing over all possible states $k,l$, accounting for the probability of having seen the observations no matter the (next) state of time $t$.

In the third step, the maximization step, we update the parameters $\theta=(A,B,\pi)$ using the estimated probabilities from the previous step.  
We compute the values for $A$ by:  
$$a_{i,j}=\frac{\sum_{t}^{T-1}\xi_t(i,j)}{\sum_{t}^{T-1}\gamma_t(i)}$$  
So this is the ratio between every time we were in state $i$ AND when the next state was $j$, to every time we were in state $i$.  
We then compute the values for $B$ by:  
$$b_{i,k}=\frac{\sum_{t}^T\gamma_t(i)(1\text{ if }O_t=k\text{ else }0)}{\sum_{t}^T\gamma_t(i)}$$  
So this is the ratio between every time we were in state $i$ AND when we observed observation $k$ at time $t$, to every time we were in state $i$.

We then repeat this procedure by going back to step 2. The algorithm will converge to a local extremum, but not necessarily the global optimum.  
We can also compute the log-likelyhood of the system: $\text{log}\sum_i\alpha_T(i)$. This will give a measure of certainty of the estimation of the parameters. This represents the log of the fraction of observation data that gets explained by the estimated parameters. The algorithm will converge to a certain value of this log-likelyhood and we detect how much this changes during every iteration step.  
My implementation for the weather HMM will run the algorithm multiple times and then store every log-likelyhood and parameter instance estimated, and then compare.

After every time we run the training, meaning we exit if the desired error tolerance is reached or if too many attempts have been used, I test the model on "future" data (not really future, since it's all synthetic), in two different ways.

I first assume the model has access to the future observation sequence, and I will test the model on its ability to generalize on estimating the underlying state sequence.  
For this, I store two structures $\delta,\psi\in\mathbb{R}^{T\times N}$, where $\delta_t(i)$ represents the maximum probability, over all possible state sequences, of seeing these states under the respected observation states, where the state at time $t-1$ ends in $i$, given the estimated parameters. So    
$$\delta_t(i)=\underset{S_0,...,S_{t-1}}\max\mathbb{P}(S_0,S_1,...,S_{t-1}=i,O_1,...,O_T\ |\ \theta)$$  
I initialize $\delta_1(i)=\pi_ib_i(O_1)$ and then run over all future timestamps $t=T+1$ until $t=T_{\text{tend}}$, where I calculate dynamically:  
$$\delta_t(i)=\text{max}\underset{j}(\delta_{t-1}(j)a_{i,j})b_i(O_t)$$  
So we compute the maximum probability of the previous timestamp multiplied by the transition probability, multiplied by the probability that we observed $O_t$.  
We also store $\psi_t(i)=\text{argmax}\underset{j}(\delta_{t-1}(j)a_{i,j})$, the state $j$, which given that the maximum probability path ended in state $i$, is the most likely to have preceded state $i$ at time $t-1$.  
Next, we initialize the predicted state sequence by setting $\text{states}_{T_\text{end}}=\text{argmax}\underset{j}\delta_{T_{\text{end}}}(j)$, and then iterate backwards:  
$$\text{states}_t=\psi_{t+1}(\text{states}_{t+1})$$  


Then I made an implementation in "baum_welch_on_financials" where I'm using Baum Welch to make a simulated investment portfolio out of synthetic data. What's interesting is that the formula for $B$ changes quite a lot, and more computation is needed for some values. I explore and explain what changes in the notebook itself.
  
