# import priors from scipy
import scipy.stats as s 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import multiprocessing as mp
import statsmodels.tsa.stattools as st

# set seed
np.random.seed(42)


# Use this function to generate parameters for the SIR model
# The arguments of the pdf are chosen to have plausible priors for beta, gamma
# NB: sample only once here

def sample_prior(exponen_scale = 0.1, beta_a = 1e-2, beta_b = 1):

    beta  = s.expon.rvs(scale=exponen_scale) # rate of infection
    gamma = s.beta.rvs(a=beta_a,b=beta_b) # prob of recovery

    return beta,gamma


# function to evolve the system given the initial conditions and the parameters beta, gamma

def sample_likelihood_and_evolve(beta=.1,gamma=.01,N_population=1000,T=600,S=None,I=10,R=0):
    
    if S == None:
        S = N_population - I - R

    S_list = np.array([S])
    I_list = np.array([I])
    R_list = np.array([R])

    # simulate the SIR model
    for _ in range(T):
            
        fraction_infected = I / N_population # fraction of population infected. Called P_{t-1} in the paper
        dI = s.binom.rvs(n=S,p=1-np.exp(-beta*fraction_infected),size=1) # number of new infections
        dR = s.binom.rvs(n=I,p=gamma,size=1) # number of recoveries

        S -= dI      # update the number of susceptible
        I += dI - dR # update the number of infected
        R += dR      # update the number of recovered

        S_list = np.append(S_list,S)
        I_list = np.append(I_list,I)
        R_list = np.append(R_list,R)

        if S + I + R != N_population:
            raise ValueError('The sum of S, I, and R should be equal to the total population size')

    return S_list,I_list,R_list
   


def sample_likelihood_and_evolve_optimized(beta=0.1, gamma=0.01, N_population=1000, T=600, S=None, I=10, R=0):
    
    if S is None:
        S = N_population - I - R

    S_list = np.empty(T+1)
    I_list = np.empty(T+1)
    R_list = np.empty(T+1)

    fraction_infected = np.zeros(T+1)
    dI = np.zeros(T+1)
    dR = np.zeros(T+1)

    fraction_infected[0] = I / N_population

    for t in range(T):
        dI[t] = np.random.binomial(S, 1 - np.exp(-beta*fraction_infected[t]))
        dR[t] = np.random.binomial(I, gamma)
        
        S -= dI[t]
        I += dI[t] - dR[t]
        R += dR[t]

        S_list[t] = S 
        I_list[t] = I
        R_list[t] = R
        
        fraction_infected[t+1] = I / N_population

    S_list[T] = S
    I_list[T] = I
    R_list[T] = R
    
    return S_list, I_list, R_list

def likelihood_free_rejection_sampler(n_iter, target_list, epsilon, measure="euclidean"):
    
    # target_list is a list of the observed data 
    S_obs = target_list[0]
    I_obs = target_list[1]
    R_obs = target_list[2]

    population = S_obs[0] + I_obs[0] + R_obs[0]

    saved_params = []
    saved_fantasy_data = []

    if measure == "autocorr":
        autocorr_obs_S = st.acf(S_obs, nlags=len(S_obs))
        autocorr_obs_I = st.acf(I_obs, nlags=len(I_obs))
        autocorr_obs_R = st.acf(R_obs, nlags=len(R_obs))

    for iteration in range(n_iter):

        n = 0 # to count the average number of generated params before they are accepted
        n_trials = np.array([])

        while True:

            #DEBUG
           #if n % 10 == 0: print(n)

            n += 1

            # generate theta from the prior
            beta,gamma = sample_prior()
            # generate fantasy data from the likelihood 
            S_gen,I_gen,R_gen = sample_likelihood_and_evolve_optimized(beta=beta,gamma=gamma)
            
            if measure == "euclidean":
                distance_S = np.linalg.norm(S_gen - S_obs)/population
                distance_I = np.linalg.norm(I_gen - I_obs)/population
                distance_R = np.linalg.norm(R_gen - R_obs)/population


            
            elif measure == "autocorr":
                
                autocorr_gen_S = st.acf(S_gen, nlags=len(S_gen))
                autocorr_gen_I = st.acf(I_gen, nlags=len(I_gen))
                autocorr_gen_R = st.acf(R_gen, nlags=len(R_gen))
                
                distance_S = np.linalg.norm(autocorr_gen_S - autocorr_obs_S)
                distance_I = np.linalg.norm(autocorr_gen_I - autocorr_obs_I)
                distance_R = np.linalg.norm(autocorr_gen_R - autocorr_obs_R)


            else: raise ValueError("The measure is not correct!")

            # Distance condition has to be met for all three curves separately
            if (distance_S <= epsilon) and (distance_I <= epsilon) and (distance_R <= epsilon):
        
                n_trials = np.append(n_trials, n)

                saved_params.append([beta, gamma]) # save params as nested

                if iteration in range(10): saved_fantasy_data.append([S_gen, I_gen, R_gen]) # save accepted fantasy data
                                                                                            # the shape of saved_fantasy_data is 10x3x601
                                                                                            
                break

    
    # count the average number of iterations 
    avg_trials = np.mean(n_trials)
        
    return np.array(saved_params), saved_fantasy_data, avg_trials

def correlate(x):
    result = signal.correlate(x, x, mode='full')
    return result[result.size//2:]


def algorithm3(inputs):
    n_iter, target_list, epsilon, norm_scale = inputs
    
    print(f"running alg3 with epsilon={epsilon}, norm_scale={norm_scale}, n_iter={n_iter}")
    measure = "euclidean"
    save_results = True
    # target_list is a list of the observed data 
    S_obs = target_list[0]
    I_obs = target_list[1]
    R_obs = target_list[2]

    population = S_obs[0] + I_obs[0] + R_obs[0]

   # saved_params = np.array([]) # to save the accepted parameters of the Markov chain
    saved_fantasy_data = []

    # Use Algorithm 2 to get a realisation (θ(0),z(0)) 
    # from the ABC target distribution πε(θ,z|y)
    print("initialising chain")
    init_values, _, _ = likelihood_free_rejection_sampler(n_iter=1,target_list=target_list,epsilon=epsilon,measure=measure) 
    #init_values = [[1,1]]
    print("init values", init_values)
    
    chain = init_values.copy()

    # Generate theta' from the proposal distribution q(θ'|θ(t-1))
    n_accepted = 0
    n_ratio_satisfied = 0
    n_S = 0
    n_I = 0
    n_R = 0

    # this will have 1s at the index corresponding to an accepted value, otherwise 0s
    accepted_in_time = np.zeros(n_iter)

    print("begin sampling")
    for i in tqdm(range(n_iter)):

        beta_old, gamma_old = chain[-1]
        
        # while loop used to avoid drawing negative values for beta_proposed and gamma_proposed
        while 1: 

            beta_proposed  = s.norm.rvs(loc=beta_old,scale=norm_scale) # rate of infection
            gamma_proposed = s.norm.rvs(loc=gamma_old,scale=norm_scale) # prob of recovery
            # Impose constraints on the parameters.
            # beta has to be positive since it is a rate of the poisson process
            # gamma has to be between 0 and 1 since it is a probability
            if beta_proposed >0 and gamma_proposed >= 0 and gamma_proposed <= 1: break
                
            
        # Generate z' from the likelihood p(z'|θ')
        S_gen,I_gen,R_gen = sample_likelihood_and_evolve_optimized(beta=beta_proposed,gamma=gamma_proposed)

        # Generate u from the uniform distribution on [0,1]
        u = np.random.uniform(0,1)

        # Compute the acceptance probability
        # pi(θ',z'|y) = πε(θ',z'|y) / q(θ'|θ(t-1))
        # 
        ratio_beta  = (s.expon.pdf(beta_proposed,scale=0.1) * s.norm.pdf(beta_old,loc=beta_proposed,scale=norm_scale)) / \
                      (s.expon.pdf(beta_old,scale=0.1)      * s.norm.pdf(beta_proposed,loc=beta_old,scale=norm_scale)) 
        

        ratio_gamma = (s.beta.pdf(gamma_proposed,a=1e-2,b=1) * s.norm.pdf(gamma_old,loc=gamma_proposed,scale=norm_scale)) / \
                      (s.beta.pdf(gamma_old,a=1e-2,b=1)      * s.norm.pdf(gamma_proposed,loc=gamma_old,scale=norm_scale))
        
        if measure == "euclidean":
            distance_S = np.linalg.norm(S_gen - S_obs)/population
            distance_I = np.linalg.norm(I_gen - I_obs)/population
            distance_R = np.linalg.norm(R_gen - R_obs)/population

        elif measure == "autocorr":

            autocorr_obs_S = st.acf(S_obs, nlags=len(S_obs))
            autocorr_obs_I = st.acf(I_obs, nlags=len(I_obs))
            autocorr_obs_R = st.acf(R_obs, nlags=len(R_obs))
            
            autocorr_gen_S = st.acf(S_gen, nlags=len(S_gen))
            autocorr_gen_I = st.acf(I_gen, nlags=len(I_gen))
            autocorr_gen_R = st.acf(R_gen, nlags=len(R_gen))
            
            distance_S = np.linalg.norm(autocorr_gen_S - autocorr_obs_S)
            distance_I = np.linalg.norm(autocorr_gen_I - autocorr_obs_I)
            distance_R = np.linalg.norm(autocorr_gen_R - autocorr_obs_R)

        
        else: raise ValueError("The measure is not correct!")
        
        # The folloing if are used to check if the conditions are satisfied
        # Just for debugging purposes
        if (u <= ratio_beta * ratio_gamma):
            n_ratio_satisfied += 1

        if (distance_S <= epsilon):
            n_S += 1

        if (distance_I <= epsilon):
            n_I += 1
        
        if (distance_R <= epsilon):
            n_R += 1


        # Accept or reject
        # Considering the product between the two ratio since the two are independent
        if (u <= ratio_beta * ratio_gamma) and (distance_S <= epsilon) and (distance_I <= epsilon) and (distance_R <= epsilon):
            
            n_accepted += 1
            accepted_in_time[i] += 1

            chain = np.vstack((chain,np.array([beta_proposed, gamma_proposed])))
            saved_fantasy_data.append([S_gen, I_gen, R_gen])
            
        else:
            chain = np.vstack((chain,np.array([beta_old, gamma_old])))

        """
        # early stopping
        if i > 1000 and np.sum(accepted_in_time[i-10:i]) == 0:
            print(f"Early stopping at iteration {i}")
            break
        """
        
        


    # A nice thing would be if data would not be overwritten (need to implement a check)
    if save_results:
        np.save(f'results/new_autocorr/chain/chain_epsilon={epsilon}_{measure}_norm={norm_scale}_Niter={n_iter}.npy', chain)
        np.save(f'results/new_autocorr/fantasy_data/fantasy_data_epsilon={epsilon}_{measure}_norm={norm_scale}_Niter={n_iter}.npy', saved_fantasy_data)
        np.save(f"results/new_autocorr/accepted_in_time/accepted_in_time_epsilon={epsilon}_{measure}_norm={norm_scale}_Niter={n_iter}.npy",np.array(accepted_in_time))

    print("ACCEPTED PROPOSALS:", n_accepted)
    print("RATIO SATISFIED:", n_ratio_satisfied)
    print("S SATISFIED:", n_S)
    print("I SATISFIED:", n_I)
    print("R SATISFIED:", n_R)

    # plotting and saving the chain
    #plot_chain(chain, epsilon, norm_scale, n_iter,measure)

    return chain, saved_fantasy_data    

def plot_chain(chain, epsilon, norm_scale, n_iter,measure):

    fig, (ax1, ax2, ax3, ax4,) = plt.subplots(4, 1, figsize=(8, 8))

    ax1.plot(chain[:,0], label='beta', color='blue', alpha=0.7)
    ax1.set_ylabel('beta')
    ax1.legend()

    ax2.plot(chain[:,1], label='gamma', color='red', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('gamma')
    ax2.legend()

    ax3.hist(chain[:,0], bins=30, density=True, alpha=0.5, label='beta')
    ax3.set_xlabel('Parameter value')
    ax3.set_ylabel('Density')
    ax3.legend()

    ax4.hist(chain[:,1], bins=30, density=True, alpha=0.5, label='gamma')
    ax4.set_xlabel('Parameter value')
    ax4.set_ylabel('Density')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'results/new_autocorr/plots/chain_epsilon={epsilon}_{measure}_norm={norm_scale}_Niter={n_iter}.pdf')

def parallel_mcmc(n_iter, target_list, epsilon_list, norm_scale_list):

    ### MULTIPROCESSING ###
    pool = mp.Pool()

    # submit multiple tasks to the pool
    
    future_results = pool.map_async(
        algorithm3, 
        [(n_iter, target_list, i, j) for i in epsilon_list for j in norm_scale_list])
    
    results = future_results.get()

    pool.close()

    return results

if __name__ == "__main__":

    import sys
    """
    try:
        epsilon = float(sys.argv[1])
        norm_scale = float(sys.argv[2])
        n_iter = int(sys.argv[3])

    except:
        epsilon = 30
        norm_scale = 0.1
        n_iter = 1000
    """

    #print(f"running alg3 with epsilon={epsilon}, norm_scale={norm_scale}, n_iter={n_iter}")   

    epsilon_values = [5,10,20]
    norm_scale_values = [0.1]

    test = sample_likelihood_and_evolve_optimized()
    results = parallel_mcmc(n_iter=1_000_000, target_list=test, epsilon_list=epsilon_values, norm_scale_list=norm_scale_values)

    # save test data
    np.save("results/euclid/target_data_14-04-1000.npy", test)

    print("PLOTS NOT SAVED; UNCOMMENT THE PLOTTING FUNCTION IN algorithm3() TO SAVE THEM")