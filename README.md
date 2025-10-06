## Expected Maximization algorithm and Baum-Welch algorithm

When reading about arbitrary optimalization and variations of the maximum-likelyhood method I stumbled accross the EM and Baum Welch algorithms, and I thought I'd experiment with them and see how they work.

In this repository I have made three small applications using these, from a very simple dice bias-parameter system to an investment portfolio implementing the Baum Welch algorithm to try to estimate market regime parameters.

### Simple EM algorithm implementation

In the file "dice.ipynb" I first explore the core workings of the EM algorithm, implemented on a simple situation where our goal is to determine the bias parameters of two dice, given a sequence of head-tail results. The problem is however, we don't know which sequence corresponds to which dice.

### Baum-Welch applied to a weather model

In the folder "WeatherModels" I then explore and implement the Baum-Welch algorithm, applied to a weather HMM. I first make a simpler implementation where we have two states and a discrete observation space, and then extend this idea into a continuous version where we have four states and a continuous observation space.

### The Baum-Welch algorithm for constructing an investment portfolio

I also made an implementation in "baum_welch_on_financials" where I'm using Baum Welch to make a simulated investment portfolio out of synthetic data.