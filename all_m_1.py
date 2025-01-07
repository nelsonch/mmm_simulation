######## Changed by Nelson Chen on July 23th, 2020

import math
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from timeit import default_timer as time
from scipy.linalg import block_diag
from scipy.special import erf
from MaxLikelihood_1g_2024 import LK_MLE_2024
import os    
warnings.simplefilter(action='ignore')

#
print(os.getcwd())
def channel_one(w = 52, L=5, alph = 0.6, lamb = 1, beta1 = 1.5, true_response = 'Weibull_S', scale = 'constant', ites = 200, Const = 500):
    ##### get the diag chunk
    def power_chen(alpha, k):
        seq_l = np.linspace(k, 0, k+1)
        tem = alpha**seq_l
        return tem
    
    # Step 1: Simulate data 
    np.random.seed(2024)
    spend = np.random.normal(1000, 100, w)
    # the spend_st is the observed spend
    seasonality = 100 * np.sin(2 * math.pi * np.arange(w) / 52) 
    #
    trend = 1*np.arange(w)
    #
    n = len(spend)
    #L = 13
    spend_s = np.zeros(n)
    for i in range(len(spend)):
        if i < L:
            # subset the first i spend
            spend_sub = spend[:i+1]
            #power_chen(alpha = 0.5, i) * spend_sub
            spend_s[i] = np.sum(power_chen(alpha = alph, k = i) * spend_sub)
        else:
            spend_sub = spend[i-L:i+1]
            spend_s[i] = np.sum(power_chen(alpha = alph, k = L) * spend_sub)
    # response 
    # let's do a scaling on spend_s
    if scale == 'constant':
        spend_s1 = spend_s/Const
    elif scale == 'mean':
        spend_s1 = spend_s/spend_s.mean()
    elif scale == 'min_max':
        spend_s1 = (spend_s - spend_s.min())/(spend_s.max() - spend_s.min())
    else:
        raise ValueError('The scaling method is not supported')
    #
    # now let's do a Hill function on spend_s
    if true_response == 'Hill_S':
        spend_trans_tempt = 1/(1 + (spend_s1/lamb)**(-2))
    elif true_response == 'Hill_C':
        spend_trans_tempt = 1/(1 + (spend_s1/lamb)**(-0.5))
    elif true_response == 'Weibull_S':
        spend_trans_tempt = 1 - np.exp(-(spend_s1/lamb)**2)
    elif true_response == 'Weibull_C':
        spend_trans_tempt = 1 - np.exp(-(spend_s1/lamb)**0.5)
    elif true_response == 'Sigmoid':
        spend_trans_tempt = 1/(1 + np.exp(-lamb*spend_s1))
    elif true_response == 'Error':
        spend_trans_tempt = erf(lamb*spend_s1)
    # scale back
    if scale == 'constant':
        spend_trans = spend_trans_tempt*Const
    elif scale == 'mean':
        spend_trans = spend_trans_tempt*spend_s.mean()
    elif scale == 'min_max':
        spend_trans = spend_trans_tempt*(spend_s.max() - spend_s.min()) + spend_s.min()
    else:
        raise ValueError('The scaling method is not supported')
    # now we can compute y 
    # np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})
    # generate z from a uniform distribution
    np.random.seed(2025)
    z1 = np.random.uniform(100, 200, w)
    np.random.seed(2029)
    z2 = np.random.uniform(20, 80, w)
    # set seed
    np.random.seed(2025)
    y = 1 + beta1*spend_trans + 0.1*seasonality + 0.2*trend + 1*z1 + 1*z2 + np.random.normal(0, 5, w)
    # now let's compute the true ROI and the true lift
    # the total spend in that 20 weeks
    #total_spend = spend[w1:w1+20].sum()
    total_spend = spend.sum()
    # total reward in that 20 weeks
    # compute y_bar without spend
    y_bar = 1 + 0.1*seasonality + 0.2*trend + 1*z1 + 1*z2
    # compute the difference in that 20 weeks
    # diff = (y[w1:w1+20+L] - y_bar[w1:w1+20+L]).sum()
    diff = (y - y_bar).sum()
    #
    roi_true = (diff - total_spend)/total_spend
    print('the true response function is', true_response)
    print('The true ROI is', roi_true)
    #
    # let's combine spend, seasonality, trend, z and y into a pd dataframe
    df = pd.DataFrame({'spend': spend, 'seasonality': seasonality, 'trend': trend, 'z1': z1, 'z2':z2, 'y': y})
    # add one more column to the beginning of the dataframe named as cluster and all elements are 1
    df.insert(0, 'cluster', 1)
    # add one more column next to cluster named as store and all elements are 1
    df.insert(1, 'store', 1)
    # add one more column next to store named as product and all elements are 1
    df.insert(2, 'product', 1)
    df.head(5)

    # Step 2: Estimate the model
    estimate = ['Weibull_S', 'Weibull_C', 'Hill_S', 'Hill_C', 'Sigmoid', 'Error']
    roi_seq = np.zeros(len(estimate))
    for j in range(len(estimate)):
        print('The response variable is', estimate[j])
        print('The scaling method is', scale)
        # now let's do the MLE estimation
        m = 1
        nn = 4
        g = 1
        L = 5
        #
        ######### get the bounds
        mybounds = [(0,1), # alpha
                    (0,2), # lambda
                    (0,2), # beta0 
                    (0,10), # beta1 
                    (0,10), (0, 10), (0, 10), (0, 10), 
                    (0, 50)] 
        ##########
        K = ites
        t_bfgs = np.inf; t_sqp = np.inf
        op_s = np.tile(0, len(mybounds))
        op_sqp = np.tile(0, len(mybounds))
        res_bfgs = np.tile(0, len(mybounds))
        res_sqp = np.tile(0, len(mybounds))
        #
        for k in range(K):
            np.random.seed(2020+2*k)
            alpha_s = np.random.uniform(low =0.01, high = 0.99, size = m)
            # np.random.seed(2020+2*k)
            # k_s = np.random.uniform(size = m)
            np.random.seed(2020+2*k)
            lambda_s = np.random.uniform(low =0.01, high = 1.99, size = m)

            np.random.seed(2020+2*k)
            intercepts = np.random.uniform(low =0.5, high = 1.5, size = g)
            
            np.random.seed(2020+2*k)
            beta = np.random.uniform(low = 0.01, high = 1.99, size = (m+nn))
            
            np.random.seed(2020+2*k)
            v0 = np.random.uniform(low = 10, high = 50, size = 1)
            
            ######### combine
            phi0 = np.concatenate((alpha_s, lambda_s, intercepts, beta, v0))
            ######### get the object 
            tem = LK_MLE_2024(Y = df, m = m, n = nn, l = L, year = int(w/52), response = estimate[j], scale = scale, Const = Const)
            ######## BFGS 
            np.random.seed(2020+2*k)
            fit1 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'L-BFGS-B', 
                                    bounds=mybounds,
                                    options={'gtol': 1e-08,
                                            'eps': 1e-08})
            if fit1.fun < t_bfgs:
                t_bfgs = fit1.fun
                op_s = fit1.x
        
            ######## SQP 
            np.random.seed(2021+2*k)
            fit2 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'SLSQP', 
                                    bounds=mybounds,
                                    options={'ftol': 1e-08, 'eps': 1e-08}) 
            if fit2.fun < t_sqp:
                t_sqp = fit2.fun
                op_sqp = fit2.x

        ####### get the results     
        res_bfgs = op_s
        res_sqp = op_sqp

        # use res_sqp as the final result
        # Step 3: Compute the ROI
        # Now let's use the number to compute the ROI
        alpha_s = res_sqp[0]
        lambda_s = res_sqp[1]
        intercepts = res_sqp[2]
        beta = res_sqp[3]
        theta = res_sqp[4:4+nn]
        # 
        spend = df['spend'].values
        #compute the predicted y 
        n = len(spend)
        #L = 13
        spend_s = np.zeros(n)
        for i in range(len(spend)):
            if i < L:
                # subset the first i spend
                spend_sub = spend[:i+1]
                #power_chen(alpha = 0.5, i) * spend_sub
                spend_s[i] = np.sum(power_chen(alpha = alpha_s, k = i) * spend_sub)
            else:
                spend_sub = spend[i-L:i+1]
                spend_s[i] = np.sum(power_chen(alpha = alpha_s, k = L) * spend_sub)

        # 
        # let's do a scaling on spend_s
        if scale == 'constant':
            spend_s1 = spend_s/Const
        elif scale == 'mean':
            spend_s1 = spend_s/spend_s.mean()
        elif scale == 'min_max':
            spend_s1 = (spend_s - spend_s.min())/(spend_s.max() - spend_s.min())
        else:
            raise ValueError('The scaling method is not supported')
        #
        # now let's do a Hill function on spend_s
        #spend_trans = 1/(1 + (spend_s1/lambda_s)**(-2))
        #
        #spend_trans = 1 - np.exp(-(spend_s1/lambda_s)**2)
        if estimate[j] == 'Hill_S':
            spend_trans_tempt = 1/(1 + (spend_s1/lambda_s)**(-2))
        elif estimate[j] == 'Hill_C':
            spend_trans_tempt = 1/(1 + (spend_s1/lambda_s)**(-0.5))
        elif estimate[j] == 'Weibull_S':
            spend_trans_tempt = 1 - np.exp(-(spend_s1/lambda_s)**2)
        elif estimate[j] == 'Weibull_C':
            spend_trans_tempt = 1 - np.exp(-(spend_s1/lambda_s)**0.5)
        elif estimate[j] == 'Sigmoid':
            spend_trans_tempt = 1/(1 + np.exp(-lambda_s*spend_s1))
        elif estimate[j] == 'Error':
            spend_trans_tempt = erf(lambda_s*spend_s1)
        else:
            raise ValueError('The response variable is not supported')

        # scale back
        if scale == 'constant':
            spend_trans = spend_trans_tempt*Const
        elif scale == 'mean':
            spend_trans = spend_trans_tempt*spend_s.mean()
        elif scale == 'min_max':
            spend_trans = spend_trans_tempt*(spend_s.max() - spend_s.min()) + spend_s.min()
        else:
            raise ValueError('The scaling method is not supported')   
        #spend_trans = spend_trans_tempt*spend_s.mean()
        # now we can compute y 
        # generate z from a uniform distribution
        # set seed
        y = intercepts + beta*spend_trans + theta[0]*df['seasonality'].values + theta[1]*df['trend'].values + theta[2]*df['z1'].values + theta[3]*df['z2'].values
        # now let's compute the true ROI and the true lift
        # the total spend in that 20 weeks
        #total_spend = spend[w1:w1+20].sum()
        total_spend = spend.sum()
        # total reward in that 20 weeks
        # compute y_bar without spend
        y_bar = intercepts + theta[0]*df['seasonality'].values + theta[1]*df['trend'].values + theta[2]*df['z1'].values + theta[3]*df['z2'].values
        # compute the difference in that 20 weeks
        # diff = (y[w1:w1+20+L] - y_bar[w1:w1+20+L]).sum()
        diff = (y - y_bar).sum()
        #
        roi = (diff - total_spend)/total_spend
        print('The estimated ROI is', roi)
        roi_seq[j] = roi
    return [{'true':roi_true}, {estimate[i]: roi_seq[i] for i in range(len(estimate))}]












    

