### The Baum-Welch algorithm for portfolio construction in finance

Here the HMM will represent a financial portfolio, where the hidden states are the different market regimes (in this simulation I define three regimes; low, medium and high volatility), the observations are asset returns for $N$ amount of assets and the parameters to be optimized are the transition matrix $A$, containing probabilities of moving from one regime to another, the initial state distribution $\pi$ which contains probabilities of starting in each state, and the emission variables $(\mu_i,\Sigma_i)$ for each regime $i$, which represent the mean returns and covariances of returns between assets.

I first implemented this approach in almost exactly the same way as I did in the weather model. However, it became very clear that some calculations were very off, and I noticed that the computations were far from numerically stable. For example, the cost matrix had values over $10^200$ and the covariance matrix $\Sigma$ became singular sometimes.
I then looked into this and found that because my data was richer and higher dimensional than in the weather system, numerical errors were accumulating all over the place.

I then decided to "normalize" some of the calculations, namely during the computation of $\alpha_t(i)$ and $\beta_t(i)$. This wasn't needed in the weather model, but here, because of the higher dimensionality of the data, it was.  
Reference: https://gregorygundersen.com/blog/2022/08/13/hmm-scaling-factors/

It roughly works like this: instead of computing

$\alpha_t(i)=b_j(O_t)\Sigma_i\alpha_{t-1}(i)a_{i,j}$

We divide by the sum of $\alpha_t$ over all states:

$\alpha_t(i)=(b_j(O_t)\sum_i\alpha_{t-1}(i)a_{i,j})/(\sum_j\alpha_t(j))$

But of course here we compute the log value of $\alpha_t(i)$ for numerical stability. We also store the value of $s_t=\sum_j\alpha_t(j)$ in a sequence called the scaling-sequence.
We now use this scaling sequence to also normalize beta:

$\beta_t(i)=(\sum_j a_{i,j}b_j(O_{t+1})\beta_{t+1}(j))/s_t$

In the calculation of $\gamma_t(i)$ and $\xi_t(i,j)$, these factors conveniently cancel out.  
The computation of the log-likelyhood is now, instead of

$\text{log-likelyhood}=\text{log}\sum_i\alpha_T(i)$

It now becomes the sum of $\text{log}\ s_t$ over all $t$, because

$\text{log-likelyhood}=\text{log}\sum_i\alpha_{T}^{\text{unscaled}}(i)=\text{log}\sum_i\alpha_{T}^{\text{scaled}}(\prod_t s_t)=\text{log}((\prod_t s_t)\sum_i\alpha_{T}^{\text{scaled}}(i))=\text{log}\prod_t s_t=\sum_t\text{log}\ s_t$

Because $\sum_i\alpha_{t}^{\text{scaled}}(i)=1$ for every $t$.





