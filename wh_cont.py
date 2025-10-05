import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


def forward(loga, logb, T, logpi, observations):
    logalpha = np.empty((T, 4))
    for i in range(4):
        logalpha[0, i] = logpi[i] + logb[0, i]
    for t in range(1, T):
        for j in range(4):
            logterms = [loga[i, j] + logalpha[t - 1, i] for i in range(4)]
            logalpha[t, j] = np.logaddexp.reduce(logterms) + logb[t, j]
        
    return logalpha

def backward(loga, logb, T):
    logbeta = np.empty((T, 4))
    logbeta[T - 1, :] = 0
    for t in range(T - 2, -1, -1):
        for i in range(4):
            logterms = [loga[i, j] + logb[t + 1, i] + logbeta[t + 1, j] for j in range(4)]
            logbeta[t, i] = np.logaddexp.reduce(logterms)
            
    return logbeta

def compute_gamma(logalpha, logbeta, T):
    loggamma = np.empty((T, 4))
    for t in range(T):
        for i in range(4):
            loggamma[t, i] = logalpha[t, i] + logbeta[t, i] - np.logaddexp.reduce([
                logalpha[t, j] + logbeta[t, j] for j in range(4)
            ])

    return loggamma

def compute_xi(logalpha, logbeta, loga, logb, observations, T):
    xi = np.empty((T, 4, 4))
    for t in range(T - 1):
        for i in range(4):
            for j in range(4):
                logterms = []
                for k in range(4):
                    for l in range(4):
                        logterms.append(
                            logalpha[t, k] + loga[k, l] + logb[t + 1, l] + logbeta[t + 1, l]
                        )
                xi[t, i, j] = (
                    logalpha[t, i] + loga[i, j] + logb[t + 1, j] + logbeta[t + 1, j]
                    - np.logaddexp.reduce(logterms)
                )
    return xi

def compute_a(loggamma, logxi, T):
    loga = np.empty((4, 4))
    for i in range(4):
        for j in range(4):
            loga[i, j] = (
                np.logaddexp.reduce([logxi[t, i, j] for t in range(T - 1)])
                - np.logaddexp.reduce([loggamma[t, i] for t in range(T - 1)])
            )
    return loga


def compute_b(mus, Sigmas, dataset, T):
    """
    mus: M x N mean returns (over time) per regime and asset
    Sigmas: M x N x N the covariance matrix per regime
    dataset: T x N matrix of returns

    returns: logB, T x M, where logB[t,k] = log p(r_t | s_t = k)
    """

    logb = np.empty((T, 4))
    for t in range(T):
        for k in range(4):
            # if np.isnan(dataset[t, :]).any() or np.isnan(mus[k, :]).any() or np.isnan(Sigmas[k, :, :]).any():
            #     raise AssertionError()
            logb[t, k] = multivariate_normal.logpdf(dataset[t, :], mus[k, :], Sigmas[k, :, :])
    return logb


def compute_mus_sigmas(loggamma, dataset, T, eps=1e-04):
    mus = np.empty((4, 2))
    Sigmas = np.empty((4, 2, 2))
    for m in range(4):
        exp_gammas = np.empty(T)

        # compute mus
        exp_gammas = np.exp(loggamma[:, m])
        den = exp_gammas.sum()
        if den < eps:
            # state got no responsibility -> reinitialize it
            mus[m, :] = np.mean(dataset, axis=0) + np.random.normal(0, 0.01, size=2)
            Sigmas[m, :, :] = np.cov(dataset.T) + eps * np.eye(2)
            continue
        
        mus[m, :] = np.sum(exp_gammas[:, None] * dataset, axis=0) / den

        # compute Sigmas
        diffs = dataset - mus[m, :]
        Sigmas[m, :, :] = (diffs.T @ (diffs * exp_gammas[:, None])) / den

        # stabilize
        Sigmas[m, :, :] = (Sigmas[m, :, :] + Sigmas[m, :, :].T) / 2
        Sigmas[m, :, :] += eps * np.eye(2)

    return mus, Sigmas


def viterbi(logpi, logb, loga, observations, T):
    logdelta = np.zeros((T, 4))
    psi = np.zeros((T, 4))
    # logdelta[t, j] = log probability of the best path ending in state j at time t
    # psi[t, j] = state that maximizes prob of having been there, over all seq
    # ended in j at time t
    logdelta[0, :] = logpi + logb[0, :]
    for t in range(1, T):
        for j in range(4):
            seq_probs = logdelta[t - 1, :] + loga[:, j]
            logdelta[t, j] = np.max(seq_probs) + logb[t, j]
            psi[t, j] = np.argmax(seq_probs)

    states = np.zeros(T, dtype=int) # holds most likely states for each time
    states[-1] = np.argmax(logdelta[-1, :])
    p = np.max(logdelta[-1, :]) # highest probability over all states (ended in T)

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
        # S_{T-1}=argmax over all j of (logdelta[T - 1, j])

    return states, p

def match_states_gaussian(mus_trained, Sigmas_trained, mus_real, Sigmas_real):
    M = len(mus_trained)
    cost_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            diff = mus_trained[i] - mus_real[j]
            cost_matrix[i, j] = np.linalg.norm(diff)  # or KL divergence
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind  # permutation mapping

def permute_model(pi, A, B, perm):
    # perm maps estimated-index -> true-index
    # We return model reordered so index i now corresponds to true index perm[i].
    # To compare, we need inverse perm that gives mapping: new_index -> old_index
    # Simpler: build arrays aligned to true indices
    N = len(perm)
    pi_reordered = np.zeros_like(pi)
    A_reordered = np.zeros_like(A)
    B_reordered = np.zeros_like(B)
    for est_i, true_i in enumerate(perm):
        pi_reordered[true_i] = pi[est_i]
    for est_i, true_i in enumerate(perm):
        for est_j, true_j in enumerate(perm):
            A_reordered[true_i, true_j] = A[est_i, est_j]
    for est_i, true_i in enumerate(perm):
        B_reordered[true_i] = B[est_i]
    return pi_reordered, A_reordered, B_reordered



import numpy as np

T = 200
T_test = 100

A_std_noise = 0.05
pi_std_noise = 0.05
mus_std_noise = [2.0, 0.5] # temp, rain level
Sigmas_std_noise = 0.02

# columns and rows indexed by
# (warm & rain, warm & dry, cold & rain, cold & dry)

# M x M
A_real = np.array([
    [0.6, 0.3, 0.05, 0.05],
    [0.2, 0.7, 0.05, 0.05],
    [0.1, 0.1, 0.7, 0.1],
    [0.05, 0.05, 0.2, 0.7]
    ], dtype=np.float32)

# M x N
mus_real = np.array([
    [20, 5],    # Warm & Rain
    [22, 0],    # Warm & Dry
    [5, 7],     # Cold & Rain
    [3, 0]      # Cold & Dry
], dtype=np.float32)

# M x N x N
Sigma_real = np.array([
    [[3, 0], [0, 2]],
    [[2, 0], [0, 1]],
    [[4, 0], [0, 3]],
    [[3, 0], [0, 1]]
], dtype=np.float32)

# M
pi_real = np.full(4, 0.25)

observations_total = np.zeros((T + T_test, 2))
states_total = np.zeros(T + T_test, dtype=int)

states_total[0] = int(np.random.choice(4))
observations_total[0, :] = np.random.multivariate_normal(mus_real[states_total[0], :], Sigma_real[states_total[0], :, :])

for t in range(1, T + T_test):
    # sample next state according to transition probabilities from A_real
    states_total[t] = np.random.choice(4, p=A_real[states_total[t-1], :])
    observations_total[t] = np.random.multivariate_normal(mus_real[states_total[t], :], Sigma_real[states_total[t], :, :])

# split data in training and testing set
observations_train = observations_total[:T, :]
observations_test = observations_total[T:, :]
states_train = states_total[:T]
states_test = states_total[T:]

print('states:'); print(states_total)
print()
print('observations:'); print(observations_total)


ll_tol = 1.0
n_attempts = 100
max_attempt_per_iter = 10

# these will store data from every run
# structure: list of lists (inner list = data per iteration)
ll_list = []
diff_norms_A = []
diff_norms_B = []
diff_norms_pi = []

# these will store data from every run
# stucture: list of floats (one per attempt, not one per iteration)
pct_states_matched_viterbi = []
frac_correct_states_predicted_list = []
frac_correct_obs_predicted_list = []

# initialize parameters for estimation (pretend we don't know the true values)
A = A_real.copy()
pi = pi_real.copy()

for attempt in range(n_attempts):

    prev_ll = 0
    log_ll = 2.0
    
    # add noise to A
    A += np.random.normal(0, A_std_noise, (4, 4))
    A = np.clip(A, a_min=0.05, a_max=0.95)
    A_rowsum = A.sum(axis=1)
    for i in range(4):
        A[:, i] /= A_rowsum
    loga = np.log(A)

    # add noise to pi
    pi += np.random.normal(0, pi_std_noise, 4)
    pi = np.clip(pi, a_min=0.05, a_max=0.95)
    pi /= pi.sum()
    logpi = np.log(pi)

    # add noise to mus
    mus = np.empty(mus_real.shape)
    mus[:, 0] = np.clip(mus_real[:, 0] + np.random.normal(0, mus_std_noise[0], size=4), a_min=0, a_max=30)
    mus[:, 1] = np.clip(mus_real[:, 1] + np.random.normal(0, mus_std_noise[1], size=4), a_min=0, a_max=30)
    # add noise to Sigma
    Sigma = Sigma_real.copy()
    for m in range(4):
        Sigma[m, :, :] = np.clip(Sigma_real[m, :, :] + Sigmas_std_noise * np.eye(2), a_min=0, a_max=10)

    # compute logB
    logb = compute_b(mus, Sigma, observations_train, T)

    # store some metrics for each iterion step
    this_diff_norms_A = []
    this_diff_norms_B = []
    this_diff_norms_pi = []
    this_attempt_ll_list = []

    this_attempt_count = 0

    while abs(prev_ll - log_ll) > ll_tol and this_attempt_count < max_attempt_per_iter:
        print('dist:', prev_ll - log_ll)
        prev_ll = log_ll

        # E step
        logalpha = forward(loga, logb, T, logpi, observations_train)
        logbeta = backward(loga, logb, T)
        loggamma = compute_gamma(logalpha, logbeta, T)
        logxi = compute_xi(logalpha, logbeta, loga, logb, observations_train, T)

        # M step
        loga = compute_a(loggamma, logxi, T)
        mus, Sigmas = compute_mus_sigmas(loggamma, observations_train, T)
        logb = compute_b(mus, Sigmas, observations_train, T)
        logpi = loggamma[0, :]

        # storing metrics for each iteration
        log_ll = np.logaddexp.reduce(logalpha[-1, :])
        this_attempt_ll_list.append(log_ll)

        this_attempt_count += 1

    A = np.exp(loga)
    A /= A.sum(axis=1, keepdims=True)
    B = np.exp(logb)
    B /= B.sum(axis=1, keepdims=True)
    pi = np.exp(logpi)
    pi /= pi.sum()

    A = np.clip(A, 1e-10, 1)
    B = np.clip(B, 1e-10, 1)

    # store data from this attempt
    ll_list.append(this_attempt_ll_list)
    diff_norms_A.append(this_diff_norms_A)
    diff_norms_B.append(this_diff_norms_B)
    diff_norms_pi.append(this_diff_norms_pi)

    # test model on training data

    perm = match_states_gaussian(mus, Sigma, mus_real, Sigma_real)
    pi_p, A_p, B_p = permute_model(pi, A, B, perm)

    print('B_p:'); print(B_p)
    print('A_p:'); print(A_p)
    print('pi_p:'); print(pi_p)

    logpi_p = np.log(pi_p)
    loga_p = np.log(A_p)
    logb_p = np.log(B_p)

    # assuming we have access to the future observations, we will test the model using the viterbi path
    # algorithm, to see how well it can predict the hidden state variables, given these observations
    predicted_states_vit, p = viterbi(logpi_p, logb_p, loga_p, observations_test, T_test)
    pct_matched = 1 - np.count_nonzero(np.logical_xor(predicted_states_vit, states_test)) / len(predicted_states_vit)
    pct_states_matched_viterbi.append(pct_matched)

    # assuming we are at time T and we wanted to predict the future states and observations up untill
    # t = T_test, we will try to predict these values and see how well the model generalizes
    predicted_obs = np.empty((T_test, 2))
    predicted_states = np.empty(T_test)

    states_p_dist = np.exp(loggamma[-1, :])
    predicted_states[0] = np.argmax(states_p_dist)
    predicted_obs[0] = np.argmax(B_p[int(predicted_states[0]), :])

    # generate prediction states and observations using A and B
    for k in range(1, T_test):

        # prediction of states
        states_p_dist = states_p_dist @ A_p
        states_p_dist /= states_p_dist.sum()
        predicted_states[k] = np.argmax(states_p_dist)
        # determine the predicted observation by taking the most likely observation, given the
        # predicted state value
        predicted_obs[k, :] = pi @ mus

    # save the fraction of correctly predicted states and observations
    frac_correct_states_predicted = 1 - np.count_nonzero(np.logical_xor(states_test, predicted_states)) / len(predicted_states)
    frac_correct_obs_predicted = 1 - np.count_nonzero(np.logical_xor(observations_test, predicted_obs)) / len(predicted_obs)

    frac_correct_states_predicted_list.append(frac_correct_states_predicted)
    frac_correct_obs_predicted_list.append(frac_correct_obs_predicted)
