from __future__ import division
import math
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from timeit import default_timer as time
from scipy.linalg import block_diag
from scipy.special import erf
warnings.simplefilter(action='ignore')
#from scipy.optimize import dual_annealing

# Function to generate AR(1) process
def simulate_ar1(mean, sd, weeks, ar_coefficient):
    ar1_process = np.zeros(weeks)
    ar1_process[0] = np.random.normal(mean, sd)
    for t in range(1, weeks):
        ar1_process[t] = mean + ar_coefficient * (ar1_process[t-1] - mean) + np.random.normal(0, sd)
    return ar1_process

# Function to generate AR(2) process
def simulate_ar2(mean, sd, weeks, ar_coefficients):
    ar2_process = np.zeros(weeks)
    ar2_process[0] = np.random.normal(mean, sd)
    ar2_process[1] = np.random.normal(mean, sd)
    for t in range(2, weeks):
        ar2_process[t] = mean + \
                         ar_coefficients[0] * (ar2_process[t-1] - mean) + \
                         ar_coefficients[1] * (ar2_process[t-2] - mean) + \
                         np.random.normal(0, sd)
    return ar2_process


##### get the diag chunk
def power_chen(alpha, k):
    seq_l = np.linspace(k, 0, k+1)
    tem = alpha**seq_l
    return tem

def adstock(x, L = 13, decay = 0.5):
    #spend_s = np.zeros((w, num_channels))
    #
    w = len(x)
    x_new = np.zeros(w)
    for i in range(w):
        if i < L:
            spend_sub = x[0:i+1]
            x_new[i] = np.sum(power_chen(alpha = decay, k = i) * spend_sub)
        else:
            spend_sub = x[i-L:i+1]
            x_new[i] = np.sum(power_chen(alpha = decay, k = L) * spend_sub)
    return x_new


##### respone function
def apply_response(x, a, b, response):
    # do the normalization of x here 
    #normalized_series = (x - x.min()) / (x.max() - x.min())
    normalized_series = x.copy()
    #
    if response == 'Hill':
        tmp = (normalized_series**b)/(a**b + normalized_series**b) # a from 0 to inf and b > 1 for S-Shaped
        #tmp = 1/(1 + (normalized_series/a)**(-b)) # a from 0 to inf and b > 1 for S-Shaped
        #return 1/(1 + (normalized_series/a)**(-b)) # a from 0 to inf and b > 1 for S-Shaped 
                                #a from 0 to inf and b from 0 to 1 for C Shaped
    # elif response == 'Hill_C':
    #     return 1/(1 + (s/l)**(-0.5))
    elif response == 'Weibull':
        tmp = 1 - np.exp(-(normalized_series/a)**b) # a from 0 to inf and b > 1 for S-Shaped
        #return 1 - np.exp(-(normalized_series/a)**b)  # a from 0 to inf and b > 1 for S-Shaped 
    #a from 0 to inf and b from 0 to 1 for C Shaped
    # elif response == 'Weibull_C':
    #     return 1 - np.exp(-(s/l)**0.5)
    elif response == 'Sigmoid':
        tmp = 1/(1 + np.exp(-b*(normalized_series - a))) # a from 0 to 1 and b from 0 to inf
        #return a/(1 + np.exp(-b*normalized_series)) # a from 0 to 1 and b from 0 to inf 
    
    elif response == 'Error':
        tmp = a*erf(b*normalized_series) # a from 0 to 1 and b from 0 to inf
        #return a*erf(b*normalized_series) # a from 0 to 1 and b from 0 to inf
    else:
        raise ValueError('The response function is not supported') 
    # need to conver back as the y_exp is already in the log scale
    return tmp*normalized_series


def transform_series(series, L, decay, a, b, assumed_response):
    adstocked = adstock(series, L, decay)
    transformed = apply_response(adstocked, a, b, assumed_response)
    return transformed


def objective_function(params, num_channels, L, nn, df, marketing_vars, control_vars, assumed_response):
    # Split params into their respective components
    decay = params[0:num_channels]  # decay for each marketing variable
    a = params[num_channels:2*num_channels]  # a for each marketing variable
    b = params[2*num_channels:3*num_channels]  # b for each marketing variable
    betas = params[3*num_channels:4*num_channels]  # beta for each marketing variable
    etas = params[4*num_channels:4*num_channels+nn]  # eta for each control variable
    intercept = params[-2]  # intercept
    error_var = params[-1]  # last parameter is error variance
    
    # Transform marketing variables
    transformed_marketing = []
    for i in range(num_channels):
        #decay, K, n, EC50 = marketing_params[4*i:4*(i+1)]
        transformed = transform_series(df[marketing_vars[i]], L, decay[i], a[i], b[i], assumed_response)
        transformed_marketing.append(transformed)
    
    # Predict y
    y_pred = sum(beta * transformed for beta, transformed in zip(betas, transformed_marketing))
    y_pred += sum(eta * df[control_var] for eta, control_var in zip(etas, control_vars))
    # add intercept
    y_pred += intercept
    # exponential transformation
    #y_pred = np.exp(y_pred)
    
    # Calculate negative log-likelihood
    residuals = df['y'] - y_pred
    n_samples = len(residuals)
    log_likelihood = -n_samples/2 * np.log(2 * np.pi * error_var) - np.sum(residuals**2) / (2 * error_var)
    return -log_likelihood  # We minimize negative log-likelihood


def generate_prediction_y(df, num_channels, L, alpha_s, a_s, b_s, beta_s, beta_price, theta, intercept, sdd, assumed_response, generate, randomseed, 
                          revenue_p_unit, cost_per_grp):
    """
    Predict spend and calculate ROI based on given parameters.

    Parameters:
    [Add parameter descriptions here]

    Returns:
    tuple: (true_roi, df) where true_roi is a list of ROIs for each channel,
           and df is the input DataFrame with an additional 'y_exp_pred' column.
    """
    # for df, only keep the first 104 rows
    df_train = df.iloc[:104, :]
    # get the last 52 weeks for testing
    df_test = df.iloc[104:, :]


    # Extract relevant data from DataFrame
    measure = df.iloc[:, 0:num_channels].values
    df_normalized = measure.copy()
    # next price 
    price = df['price'].values
    price_log = np.log(price)
    # seasonality
    seasonality = df['seasonality'].values
    #df_normalized = (spend - np.min(spend, axis=1)[:, np.newaxis]) / np.ptp(spend, axis=1)[:, np.newaxis]
    
    # z_n
    #   if y exist in the df's column name 
    # if 'y' in df.columns: 
    #     z_n = df.iloc[:, num_channels:-1].values
    # else:
    #z_n = df.iloc[:, num_channels:].values
    #z_normalized = z_n.copy()
    # check if z_n's shape[1] == nn
    #if z_n.shape[1] != nn:
    #    raise ValueError('The number of non-carryover channels should be equal to nn')
    # do the normalization of z_n
    #z_normalized = (z_n - np.min(z_n, axis=1)[:, np.newaxis]) / np.ptp(z_n, axis=1)[:, np.newaxis]

    #print(z_n.shape)
    w = df.shape[0] # 3 years maybe 156 weeks

    # Apply adstock to all channels
    measure_s = np.zeros((w, num_channels))
    #
    for channel in range(num_channels):
        for i in range(w):
            if i < L:
                measure_sub = df_normalized[0:i+1, channel]
                measure_s[i, channel] = np.sum(power_chen(alpha = alpha_s[channel], k=i) * measure_sub)
            else:
                measure_sub = df_normalized[i-L:i+1, channel]
                measure_s[i, channel] = np.sum(power_chen(alpha = alpha_s[channel], k=L) * measure_sub)
    
    # Apply response function to all channels
    #spend_trans = np.array([apply_response(df_normalized[:, channel], a_s[channel], b_s[channel], assumed_response) for channel in range(num_channels)])
    measure_trans = np.zeros_like(measure_s)
    for channel in range(num_channels):
        measure_trans[:, channel] = apply_response(measure_s[:, channel], a_s[channel], b_s[channel], assumed_response)
    #print(spend_trans.shape)
    
    # take log 
    measure_trans_copy = measure_trans.copy()
    measure_trans_copy[measure_trans <= 0] = 0.1
    measure_trans_log = np.log(measure_trans_copy)
    # column mean of measure_trans_log
    # print(np.mean(measure_trans_log, axis=0))

    # 
    # print('intercept:', intercept)
    # print('beta_price:', price_log[0:5])
    # print('measure_trans_log:', measure_trans_log[0:5, :])
    # print('beta_s:', beta_s)
    # np.sum(beta_s[np.newaxis, :] * measure_trans_log, axis=1)
    # print('seasonality:', seasonality[0:5])

    # Generate response variable
    np.random.seed(randomseed)
    if generate:
        y = (intercept + beta_price * price_log +  
            np.sum(beta_s[np.newaxis, :] * measure_trans_log, axis=1) + 
            theta * seasonality + np.random.normal(0, sdd, w))
        #print('y:', y[0:5])
        df['y'] = np.exp(y)
    else:
        y = (intercept + beta_price * price_log +  
            np.sum(beta_s[np.newaxis, :] * measure_trans_log, axis=1) + 
            theta * seasonality)
        df['y_hat'] = np.exp(y)

    # except the last column, take np.exp() of the columns to convert back. 
    #df.iloc[:, :-1] = np.exp(df.iloc[:, :-1])
    #
    #y = np.exp(y)
    # Calculate ROI for each channel
    estimate_roi = []
    #y_tmp = np.exp(y) # same for both cases 
    #
    # compute ROI only using the df_test
    # get the last 52 weeks for price_log and seasonality
    price_log_test = price_log[104:]
    seasonality_test = seasonality[104:]
    # get the last 52 weeks for measure
    measure_trans_log_test = measure_trans_log[104:, :]
    #
    all = np.sum(np.exp(intercept + beta_price * price_log_test +
                        np.sum(beta_s[np.newaxis, :] * measure_trans_log_test, axis=1) + 
                        theta * seasonality_test))
    
    for channel in range(num_channels):
        #print(channel)
        # without channel 1
        tmp_without = np.sum(np.exp(y[104:] - beta_s[channel] * measure_trans_log_test[:, channel]))
        #y_without = y - beta_s[channel]*spend_trans[:, channel]
        diff = all - tmp_without
        # revenue by converntion 
        diff_revenue = diff * revenue_p_unit[channel]
        #diff = np.sum(beta_s[channel] * spend_trans[:, channel])
        #total_grp = np.sum(df[:, channel])
        total_grp = np.sum(df_test.iloc[:, channel])
        total_grp_cost = total_grp * cost_per_grp[channel]
        # compute roi 
        roi = (diff_revenue - total_grp_cost) / total_grp_cost
        estimate_roi.append(roi)

    # #print(estimate_roi)
    # #print(df.head(5))
    return df, estimate_roi
    #return df


def simulate_data(w = 156, L = 13, num_channels = 4, true_response = 'Hill', randomseed = 2024, 
                  ar12 = ['ar1', 'ar2', 'ar1', 'ar2'], 
                  ar_mean = [500, 1500, 2000, 1000], 
                  ar_sd = [10, 10, 10, 10, 10, 10], 
                  ar_coefficients = {'ar1': [0.7], 'ar2': [0.3, 0.5]},
                  alpha_s = None, a_s = None, b_s = None,
                  beta_s = None, beta_price = None, theta = None, 
                  intercept = None, sdd = None, 
                  revenue_p_unit = None, cost_per_grp = None
                  ):
    # if nn <2 raise error
    # if nn < 1:
    #     raise ValueError('The number of non-carryover channels should be at least 1, seasonality')
    
    # convert alpha, a, b, beta, theta from list to numpy arrays
    alpha_s = np.array(alpha_s)
    a_s = np.array(a_s)
    b_s = np.array(b_s)
    beta_s = np.array(beta_s)
    #theta = np.array(theta)
    
    # Simulate data for multiple channels
    simulated_data = np.zeros((w, num_channels))
    for i, ar_type in enumerate(ar12):
        mean = ar_mean[i]
        sd = ar_sd[i]
        if ar_type == 'ar1':
            ar_coefficient = ar_coefficients['ar1'][0]
            np.random.seed(randomseed+i)
            simulated_data[:, i] = simulate_ar1(mean, sd, w, ar_coefficient)
        elif ar_type == 'ar2':
            ar_coefficients_ar2 = ar_coefficients['ar2']
            np.random.seed(randomseed+i)
            simulated_data[:, i] = simulate_ar2(mean, sd, w, ar_coefficients_ar2)
        else:
            raise ValueError('Unsupported AR type')
    simulated_df = pd.DataFrame(simulated_data, columns=['channel_' + str(i) for i in range(num_channels)])
    
    # simulate price 
    np.random.seed(randomseed+30)
    price = np.random.normal(10, 0.3, w)
    #any price < 0 , set as 1
    price[price <= 0] = 10
    simulated_df['price'] = price

    # Simulate data for non-carryover channels
    seasonality = 100 * np.sin(2 * math.pi * np.arange(w) / 52)
    # mean shift of 100
    seasonality += 200
    # convert seasonality to DataFrame
    seasonality_df = pd.DataFrame(seasonality, columns=['seasonality'])
    #trend = 1*np.arange(w)
    
    # Simulate nn-2 time series from MA series
    # ma_processes = np.zeros((w, nn-1))
    # for i in range(nn-2):
    #     np.random.seed(randomseed+4*i)
    #     ma_process = np.random.normal(np.mean(ar_mean), np.mean(ar_sd), w)
    #     ma_processes[:, i] = np.convolve(ma_process, np.ones(3)/3, mode='same')  # Simple moving average with window size 3

    # Combine seasonality, trend, and MA processes into a DataFrame
    # non_carryover_df = pd.DataFrame(ma_processes, columns=['z_' + str(i) for i in range(2, nn)])
    # non_carryover_df.insert(0, 'trend', trend)
    # non_carryover_df.insert(0, 'seasonality', seasonality)

    # Combine all data into a single DataFrame
    df_data = pd.concat([simulated_df, seasonality_df], axis=1)

    # prepare for np.log if any df_data is 0 or negative, change to 0.1
    #df_data[df_data <= 0] = 0.1
    # log transformation
    #df_data = np.log(df_data)
    # generate

    return generate_prediction_y(df = df_data, num_channels = num_channels, L = L, 
                                    alpha_s= alpha_s, a_s= a_s, b_s = b_s, 
                                    beta_s = beta_s, beta_price = beta_price, theta = theta,
                                    intercept = intercept, sdd = sdd, 
                                    assumed_response = true_response,
                                    generate = True, randomseed = randomseed, 
                                    revenue_p_unit = revenue_p_unit, cost_per_grp = cost_per_grp)


def estimate_parameters(df = None, L = 13, assumed_response='Weibull', num_channels = 6, maxiter = 500,
                        alpha_bound = (0, 1), a_bound = (0, 1), b_bound = (0, 10), 
                        beta_bound = (0, 5), theta_bound = (0, 1), intercept_bound = (0, 10), v0_bound = (0, 1)):
    
    #
    nn = int(df.shape[1] - num_channels - 1)
    #### create bounds for estimation
    #
    alpha_bounds = [alpha_bound] * num_channels
    a_bounds = [a_bound] * num_channels
    b_bounds = [b_bound] * num_channels
    # intercept:
    intercept_bounds = [intercept_bound]
    # beta
    beta_bounds = [beta_bound] * num_channels
    # theta
    theta_bounds = [theta_bound] * nn
    v0_bounds = [v0_bound]
    # combine the bounds
    mybounds = alpha_bounds + a_bounds + b_bounds + beta_bounds + theta_bounds + intercept_bounds + v0_bounds
    # DONE 
    
    #  call the optimization function
    marketing_vars = df.columns[:num_channels]
    control_vars = df.columns[num_channels:-1]
    args = (num_channels, L, nn, df, marketing_vars, control_vars, assumed_response)
    result = dual_annealing(objective_function, mybounds, args = args, maxiter = maxiter)
    
    # now get the results 
    params = result.x
    decay = params[0:num_channels]  # decay for each marketing variable
    a = params[num_channels:2*num_channels]  # a for each marketing variable
    b = params[2*num_channels:3*num_channels]  # b for each marketing variable
    betas = params[3*num_channels:4*num_channels]  # beta for each marketing variable
    etas = params[4*num_channels:4*num_channels+nn]  # eta for each control variable
    intercept = params[-2]  # intercept
    error_var = params[-1]  # last parameter is error variance
    # return the results as a dictionary
    return {'decay': decay, 'a': a, 'b': b, 'betas': betas, 'etas': etas, 'intercept': intercept, 'error_var': error_var}

    # Define bounds for parameters
    # bounds = ([(0, 1), (0, 2), (0, 10), (0, 1)] * m +  # decay, K, n, EC50 for each marketing variable
    #         [(0, 1)] * m +  # beta for each marketing variable
    #         [(0, 1)] * nn +  # eta for each control variable
    #         [(0, 100)])   # error variance

    # Run simulated annealing
    

    ##### Estimate the model 
    # K = ites
    # t_bfgs = np.inf
    # t_sqp = np.inf
    # op_s = np.tile(0, len(mybounds))
    # op_sqp = np.tile(0, len(mybounds))

    # res_bfgs = np.tile(0, len(mybounds))
    # res_sqp = np.tile(0, len(mybounds))
    # def objective_function1(x_list, df, num_channels, nn, L, assumed_response):
    #     # Extract the values from the list
    #     x = [x_list[i] for i in range(len(x_list))]
    #     # Calculate the negative log posterior
    #     tem = LK_MLE_2025(Y = df, m = num_channels, n = nn, l = L, response = assumed_response)
    #     return tem._negative_log_posterior(x)
    
    # using dual_annealing
    #gg = dual_annealing(objective_function1, mybounds, args=(df, num_channels, nn, L, assumed_response), seed=2024, maxiter=15000, atol=1e-8)
    # for k in range(K):
    #     np.random.seed(random_seed+2*k)
    #     alpha_s = np.random.uniform(low =0.01, high = 0.99, size = num_channels)
    #     if assumed_response == 'Hill' or assumed_response == 'Weibull':
    #         np.random.seed(random_seed+2*k)
    #         a_s = np.random.uniform(low =0.01, high = 10, size = num_channels)
    #         np.random.seed(random_seed+2*k)
    #         b_s = np.random.uniform(low =0.01, high = 1.99, size = num_channels)
    #     else:
    #         np.random.seed(random_seed+2*k)
    #         a_s = np.random.uniform(low =0.01, high = 0.99, size = num_channels)
    #         np.random.seed(random_seed+2*k)
    #         b_s = np.random.uniform(low =0.01, high = 1.99, size = num_channels)

    #     np.random.seed(random_seed+2*k)
    #     intercepts = np.random.uniform(low =0.5, high = 1.5, size = 1)
        
    #     np.random.seed(random_seed+2*k)
    #     beta = np.random.uniform(low = 0.01, high = 1.99, size = (num_channels + nn))
        
    #     np.random.seed(random_seed+2*k)
    #     v0 = np.random.uniform(low = 10, high = 50, size = 1)
        
    #     ######### combine
    #     phi0 = np.concatenate((alpha_s, a_s, b_s, intercepts, beta, v0))
    #     ######### get the object 
    #     tem = LK_MLE_2025(Y = df, m = num_channels, n = nn, l = L, response = assumed_response)
    #     ######## BFGS 
    #     # np.random.seed(2020+2*k)
    #     # fit1 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'L-BFGS-B', 
    #     #                         bounds=mybounds,
    #     #                         options={'gtol': 1e-08, 'eps': 1e-08})
    #     # if fit1.fun < t_bfgs:
    #     #     t_bfgs = fit1.fun
    #     #     op_s = fit1.x
    #     ######## SQP 
    #     np.random.seed(2021+2*k)
    #     fit2 = optimize.minimize(tem._negative_log_posterior, x0=phi0, method = 'SLSQP', 
    #                             bounds=mybounds,
    #                             options={'ftol': 1e-08, 'eps': 1e-07}) 
    #     if fit2.fun < t_sqp:
    #         t_sqp = fit2.fun
    #         op_sqp = fit2.x
    #     print(k)
    #     ####### get the results     
    #     # res_bfgs = op_s
    #     # res_sqp = op_sqp
    # # Now let's use the number to compute the ROI
    # # use res_sqp as the final result
    # # get the results
    # success = gg.success
    # message = gg.message
    # print('Estimation is done')
    # print(success)
    # print(message)
    # #
    # res = gg.x
    # alpha_s = res[0:num_channels]
    # a_s = res[num_channels:num_channels*2]
    # b_s = res[num_channels*2:num_channels*3]
    # intercepts = res[num_channels*3:num_channels*3+1]
    # beta = res[num_channels*3+1:num_channels*3+1+num_channels]
    # theta = res[num_channels*3+1+num_channels:-1]
    # v0 = res[-1]
    # # get the prediction and final ROI using prediction function 
    # print('Estimation is done')
    # # get a new df without the last column
    # df_X = df.iloc[:, :-1]
    # estimated_roi, df_new = prediction_spend(df_X, num_channels, L, alpha_s, a_s, b_s, intercepts, beta, theta, assumed_response, False)
    # return estimated_roi, df_new, res



# step 1, simulate data 
# step 2, estimate parameters
# step 3, predict and get the ROI



# class LK_MLE_2025(object):
    
#     def __init__(self, Y, m=2, n=1, l=5, response='Hill'):
#         self.l = l  # period the carryover effect disappears
#         self.m = m  # number of variables with carryover effects 
#         self.n = n  # number of variables without carryover effects
#         self.w = Y.shape[0]  # finest combination is assumed to have 52 weeks (one year) data
#         self.response = response  # response variable
#         #
#         self.Y = Y
#         self.Y_dat = Y.iloc[:, 0:self.m].values
#         self.Y_dat_n = Y.iloc[:, self.m:self.m+self.n].values
#         self.Y_Y = Y.iloc[:, -1].values

#     def _negative_log_posterior(self, phi):
#         """
#         Calculate the negative log posterior.
#         """
#         self._get_para(phi)
#         spend_s = self._apply_adstock()
#         spend_trans = self._apply_response_function(spend_s)
#         mu_fi = self._calculate_mu_fi(spend_trans)
#         ll = self._calculate_log_likelihood(mu_fi)
#         return -ll

#     def _apply_adstock(self):
#         """
#         Apply adstock to all channels.
#         """
#         spend = self.Y_dat.copy()
#         df_normalized = (spend - np.min(spend, axis=1)[:, np.newaxis]) / np.ptp(spend, axis=1)[:, np.newaxis]
#         num_channels = spend.shape[1]
#         spend_s = np.zeros((self.w, num_channels))
#         for channel in range(num_channels):
#             for i in range(self.w):
#                 if i < self.l:
#                     spend_sub = df_normalized[:i+1, channel]
#                     spend_s[i, channel] = np.sum(power_chen(alpha=self.alpha_s[channel], k=i) * spend_sub)
#                 else:
#                     spend_sub = df_normalized[i-self.l:i+1, channel]
#                     spend_s[i, channel] = np.sum(power_chen(alpha=self.alpha_s[channel], k=self.l) * spend_sub)
#         return spend_s

#     def _apply_response_function(self, df_normalized):
#         """
#         Apply response function to all channels.
#         """
#         num_channels = df_normalized.shape[1]
#         spend_trans = np.zeros_like(df_normalized)
#         for channel in range(num_channels):
#             spend_trans[:, channel] = apply_response(df_normalized[:, channel], self.a_s[channel], self.b_s[channel], self.response)
#         return spend_trans

#     def _calculate_mu_fi(self, spend_trans):
#         """
#         Prepare data for further calculations.
#         """
#         temp = np.zeros_like(self.Y_dat_n)
#         for col in range(self.Y_dat_n.shape[1]):
#             min_value = np.min(self.Y_dat_n[:, col])
#             max_value = np.max(self.Y_dat_n[:, col])
#             temp[:, col] = (self.Y_dat_n[:, col] - min_value) / (max_value - min_value)
        
#         # print(self.intercepts)
#         # print(self.beta_s)
#         # print(spend_trans.shape)
#         # print(spend_trans)
#         #print(self.beta_s*spend_trans)
#         #beta_s = self.beta_s
#         #print(np.sum(np.array(self.beta_s)[np.newaxis, :] * spend_trans, axis=1))
#         #print(np.sum(np.arry(self.theta_s)[np.newaxis, :]*temp, axis=1))
#         hat_y = self.intercepts + np.sum(np.array(self.beta_s)[np.newaxis, :] * spend_trans, axis=1) + np.sum(np.array(self.theta_s)[np.newaxis, :]*temp, axis=1)
#         #return np.exp(hat_y)
#         return hat_y

#     def _calculate_log_likelihood(self, mu_fi):
#         """
#         Calculate log likelihood.
#         """
#         #return self._log_normpdf_nelson(self.Y_Y, mu_fi, np.sqrt(self.variance0)).sum()
#         # minimize sum squared of errors between hat_y and y
#         #return -np.sum((self.Y_Y - mu_fi)**2)
#         #return self.Y_Y - mu_fi
#         # compute the log likelihood of the normal distribution of mu_fi
#         return -0.5 * np.sum(np.log(2 * np.pi * self.variance0) + (self.Y_Y - mu_fi)**2 / self.variance0) 

#     def _get_para(self, phi):
#         """
#         Get parameters from phi.
#         """
#         self.alpha_s = phi[0:self.m].copy()
#         self.a_s = phi[self.m:2*self.m].copy()
#         self.b_s = phi[2*self.m:3*self.m].copy()
#         self.intercepts = phi[3*self.m:3*self.m+1].copy()
#         self.beta_s = phi[3*self.m+1: 4*self.m+1].copy()
#         self.theta_s = phi[4*self.m+1:-1].copy()
#         self.variance0 = phi[-1]

#     # @staticmethod
#     # def _log_normpdf_nelson(x, mu, sigma):
#     #     """
#     #     Calculate log of normal probability density function.
#     #     """
#     #     return -0.5*np.log(2*math.pi*sigma**2)-(x-mu)**2/2/sigma**2
