######## Changed by Nelson Chen on July 23th, 2020

import math
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from timeit import default_timer as time
from scipy.linalg import block_diag
from scipy.special import erf
warnings.simplefilter(action='ignore')

class LK_MLE_2024():
    
    def __init__(self, Y = None, m = 2, n = 1, l = 5, year = 1, response = 'Hill_S', scale = 'constant', Const = 500):
        #######
        ###### priors need to be a dictionary 
        
        ### the input data is assumed a pandas data frame that have 3 columns: 'cluster', 'store', 'product'. Conisder fixed effect at cluster level and random effect at store level. The last column is assumed to have the response variable
        #######
        self.l = l ###### period the carryover effect disappears
        self.m = m ##### # of variables with carryover effects 
        self.n = n ##### # of variables without carryover effects
        self.w = 52*year ####### the finest combination is assumed to have 52 weeks (one year) data. This can be relaxed. 
        self.response = response ###### the response variable
        self.scale = scale ###### the scaling method
        self.Const = Const


        self.cluster = len(np.unique(Y['cluster'].ravel())) ### 1
        self.cluster_size = Y.groupby('cluster', sort=False).store.nunique().ravel() ###(1)
        cluster_size_0 = np.concatenate((np.array([0]), self.cluster_size)) ### (0, 1)
        self.cluster_size_sum = np.cumsum(cluster_size_0) #### (0, 1)
        self.total_size = self.cluster_size_sum[-1] ## 1
        
        ####
        self.num_product_store = Y.groupby(['cluster', 'store']).size().ravel()  ###### get the number of products per store (52)
        self.tem_index_row = np.cumsum(self.num_product_store)[:-1]
        self.tol_product_size = int((Y.groupby(['cluster', 'store']).size()/self.w).ravel().sum()) #### get the number of individual product size 1
        self.Y = Y
        
        ######################## Split into two data frames
        self.Y_dat = Y.iloc[:,3:3+self.m].values # select only the m columns and convert to np array
        self.Y_dat_n = Y.iloc[:,3+self.m:3+self.m+self.n].values #selet the n columns
        self.Y_Y = Y.iloc[:,-1].values ##### get the last column
        #######
        #self.priors = priors
        #R_squared = self._compute_R_squared(res, Y)


    def _negative_log_posterior(self, phi = None):
        """
        preserve the order for cluster, store and product
        each block returns l-1:w rows,i.e., the first l-1 rows are removed.
        """
        ############
        self._get_para(phi)
        #self.cluster = len(np.unique(Y['cluster'].ravel())) ### 2
        #self.cluster_size = Y.groupby('cluster', sort=False).store.nunique().ravel() ###(3, 3)
        #cluster_size_0 = np.concatenate((np.array([0]), self.cluster_size)) ### (0, 3, 3)
        #self.cluster_size_sum = np.cumsum(cluster_size_0) #### (0, 3, 6)
        #self.total_size = self.cluster_size_sum[-1] ## 6 stores
        ############# compute priors
        ###### manupilation on Y_dat
        Y_dat1 = np.concatenate((self.alpha_s[np.newaxis, :], self.Y_dat), 0)
        ###### apply the decay effects:
        Y_dat_after0 = np.apply_along_axis(self._decay_effects, 0, Y_dat1)
        # apply the scaling 
        if self.scale == 'constant':
            Y_dat_after0_tempt = Y_dat_after0/self.Const
        elif self.scale == 'min_max':
            Y_dat_after0_tempt = (Y_dat_after0 - Y_dat_after0.min())/(Y_dat_after0.max() - Y_dat_after0.min())
        elif self.scale == 'mean':
            Y_dat_after0_tempt = Y_dat_after0/Y_dat_after0.mean()
        else:
            raise ValueError('The scaling method is not supported')
        #####  then apply the shape and scale effects:
        if self.response == 'Hill_S':
            Y_dat_after = 1/(1 + (Y_dat_after0_tempt/self.lambda_s)**(-2))
        elif self.response == 'Hill_C':
            Y_dat_after = 1/(1 + (Y_dat_after0_tempt/self.lambda_s)**(-0.5))
        elif self.response == 'Weibull_S':
            Y_dat_after = 1 - np.exp(-(Y_dat_after0_tempt/self.lambda_s)**2)
        elif self.response == 'Weibull_C':
            Y_dat_after = 1 - np.exp(-(Y_dat_after0_tempt/self.lambda_s)**0.5)
        elif self.response == 'Sigmoid':
            Y_dat_after = 1/(1 + np.exp(-self.lambda_s*Y_dat_after0_tempt))
        elif self.response == 'Error':
            Y_dat_after = erf(self.lambda_s*Y_dat_after0_tempt)
        else:
            raise ValueError('The response variable is not supported')
        # scale back 
        if self.scale == 'constant':
            Y_dat_after = Y_dat_after*self.Const
        elif self.scale == 'min_max':
            Y_dat_after = Y_dat_after*(Y_dat_after0.max() - Y_dat_after0.min()) + Y_dat_after0.min()
        elif self.scale == 'mean':
            Y_dat_after = Y_dat_after*Y_dat_after0.mean()
        else:        
            raise ValueError('The scaling method is not supported')
        ############### need to combine
        Y_dat_after_full = np.concatenate((Y_dat_after, self.Y_dat_n), 1)

        ##### need to convert to a wide form of matrix based on store number
        #block_diag(*([a] * 6))
        #################
        tt = np.split(Y_dat_after_full, indices_or_sections=self.tem_index_row, axis=0)
        final_Y = block_diag(*(tt * 1))
        mu_nointercepts = final_Y@self.beta_s[:, np.newaxis]

        ################ the intercepts
        mu_0 = np.repeat(self.intercepts, self.num_product_store)[:, np.newaxis]
        ########### final
        mu_fi = mu_0 + mu_nointercepts
        
        ############ Get the likelihood
        #ll = norm.logpdf(Y_Y, mu_fi, np.sqrt(self.variance0)).sum()
        ll = self._log_normpdf_nelson(self.Y_Y, mu_fi, np.sqrt(self.variance0)).sum()


        ########### Now need to compute the random effects
        #beta_f = np.reshape(self.beta_s, (self.cluster, self.m+self.n))
        #beta_f_repeat = np.repeat(beta_f, self.cluster_size, axis=0).ravel()
        ###### variance
        #v_f = np.reshape(self.variance_s, (self.cluster, self.m+self.n))
        #s_f_repeat = np.sqrt(np.repeat(v_f, self.cluster_size, axis=0)).ravel()
        #############
        #ll_random = norm.logpdf(self.beta_s_random.ravel(), beta_f_repeat, s_f_repeat).sum()
        #ll_random = self._log_normpdf_nelson(self.beta_s_random.ravel(), beta_f_repeat, s_f_repeat).sum()
        #ll_random = self._log_tnpdf_nelson(self.beta_s_random.ravel(), beta_f_repeat, s_f_repeat).sum()

        ####### final likelihood
        final_results = -1*(ll + 0)

        #if(Prediction == False):
        return final_results
        #else:
        #    return [mu_fi]
            

    ############ get parameters from phi
    def _get_para(self, phi):
        ##### get the values from phi
        self.alpha_s = phi[0:self.m].copy()
        #self.alpha_s = np.exp(self.alpha_s_star)/(np.exp(self.alpha_s_star) + 1)
        
        # self.k_s = phi[self.m:2*self.m].copy()
        # self.lambda_s = phi[2*self.m:3*self.m].copy()
        self.lambda_s = phi[self.m:2*self.m].copy()

        ### fixed effects including intercepts
        self.intercepts = phi[2*self.m:2*self.m+self.total_size].copy()
        self.beta_s = phi[2*self.m+self.total_size: 2*self.m+self.total_size+(self.m+self.n)*self.cluster].copy()
        #self.variance_s = phi[3*self.m+self.total_size+(self.m+self.n)*self.cluster: 3*self.m+self.total_size+2*(self.m+self.n)*self.cluster].copy()

        ##### random effects
        #self.beta_s_random = phi[3*self.m+self.total_size+2*(self.m+self.n)*self.cluster:3*self.m+self.total_size+2*(self.m+self.n)*self.cluster+(self.m+self.n)*self.total_size].copy()
        #self.variance0 = phi[3*self.m+self.total_size+2*(self.m+self.n)*self.cluster+(self.m+self.n)*self.total_size:3*self.m+self.total_size+2*(self.m+self.n)*self.cluster+(self.m+self.n)*self.total_size+1]
        self.variance0 = phi[2*self.m+self.total_size+(self.m+self.n)*self.cluster:2*self.m+self.total_size+(self.m+self.n)*self.cluster+1]
    
    
    ########### Below 3 functions to convert X into X^{\star} taking the carryover effect into account 
    def _decay_effects(self, x_alpha):
        alpha = x_alpha[0]; x = x_alpha[1:]
        alpha_matrix = self._decay_matrix(alpha)
        tem = alpha_matrix@x.reshape(self.w, self.tol_product_size, order = 'F')
        #### reshape back  by column
        x_decay = tem.ravel(order = 'F')
        return x_decay

    
    ##### get the diag chunk
    def _power_chen(self, alpha):
        seq_l = np.linspace(self.l, 0, self.l+1)
        tem = alpha**seq_l
        return tem

  
    ##### get the decay matrix
    def _decay_matrix(self, alpha):
        a = self._power_chen(alpha)
        ####### one for self.l:self.n
        tem1 = np.zeros((self.w, self.w))
        tem2 = np.zeros((self.l+1, self.w)) ##### another for the first l rows
        for h in range(self.l+1): #### only loop l times
            tem1 = tem1 + np.eye(self.w, k=h)*a[h]
            tem2[h, 0:h+1] = a[1:][self.l-1-h:]
        #tem2 = np.zeros((self.l, self.w))
        #for jj in range(self.l):
        #    tem2[jj, 0:jj+1] = a[1:][self.l-1-jj:]
        ###### combine 
        tem_ma = np.concatenate((tem2[0:-1, :], tem1[0:self.w - self.l, :]), 0)
        return tem_ma
        #return block_diag(*([a] * self.w))

  
 
    #######4 functions for norm, gamma, inv gamma, truncated Normal log PDF 
    def _log_normpdf_nelson(self, x, mu, sigma):
        return -0.5*np.log(2*math.pi*sigma**2)-(x-mu)**2/2/sigma**2

        
        