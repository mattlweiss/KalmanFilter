import sys
import tensorflow as tf
import numpy as np
np_float_prec = np.float64
from math import log
from pdb import set_trace as st
from dovebirdia.filtering.kalman_filter import KalmanFilter,ExtendedKalmanFilter

from tensorflow.python.ops.parallel_for.gradients import jacobian

class InteractingMultipleModelKalmanFilter(KalmanFilter):

    def __init__(self,
                 meas_dims=None,
                 state_dims=None,
                 dt=None,
                 model_order=None,
                 models=None,
                 H=None,
                 R=None,
                 x0=None,
                 P0=None,
                 p=None,
                 mu=None):

        """
        Implements an Interacting Multiple Model Kalman Filter, derived from KalmanFilter class
        """

        super().__init__(meas_dims=meas_dims,
                         state_dims=state_dims,
                         model_order=model_order,
                         dt=dt,
                         H=H,
                         R=R,
                         x0=x0,
                         P0=P0)

        # IMMKF specific parameters
        self.models = models
        self.model_list = list(self.models.values())
        self.p = p
        self.mu = mu

        # normalize mu, should be normalized in config file, this is a sanity check
        self.mu /= self.mu.sum()
        
        # number of models
        self.n_models = len(self.model_list)
        
        # initialize mixing probabilities
        self.mix_prob = np.zeros((self.n_models,self.n_models))

    def fit(self, inputs):

        """
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        z = super().process_inputs(inputs)

        _, _, x_hat_pri, x_hat_post, P_pri, P_post, self._kf_ctr, mu = tf.scan(self.kfScan,
                                                                               z,
                                                                               initializer = [ [self.x0]*self.n_models,
                                                                                               [self.P0]*self.n_models,
                                                                                               self.x0, self.x0,
                                                                                               self.P0, self.P0,
                                                                                               tf.constant(0),
                                                                                               self.mu ], name='kfScan')

        filter_results = super().process_results(x_hat_pri, x_hat_post, P_pri, P_post,z, mu)

        return filter_results
    
    def kfScan(self, state, z):

        """ This is where the Kalman Filter is implemented. """

        x_post, P_post, _ , _ , _ , _ , self.kf_ctr,_ = state

        # Hold accumlated a priori and a posteriori x and P along with likelihood of measurement
        self.x_pri = list()
        self.P_pri = list()
        self.x_post = list()
        self.P_post = list()
        self.Lambda = list()

        ##################################################################
        # Estimation with Applications to Tracking and Navigation, pg. 455
        # Step 1 - Calculation of Mixing Probabilities
        ##################################################################

        # compute c_bar and \mu_{i|j}
        self.compute_mixing_probabilities()

        ######################################################################
        # Estimation with Applications to Tracking and Navigation, pg. 455-456
        # Step 2 - Mixing
        ######################################################################

        # compute mixed initial estimate and covariance
        x_post_star, P_post_star = self.compute_mixed_state_and_covariance(x_post,P_post)

        ########################
        # Run each Kalman Filter
        ########################

        # loop over models, computing a priori estimate for each 
        for self.model_index, self.model in enumerate(self.model_list): 

            #########
            # predict
            #########
            x_pri, P_pri = self.predict(x_post_star[self.model_index],P_post_star[self.model_index],self.model)

            # save since we're using this in the AEIMMKF
            self.x_pri.append(x_pri)
            self.P_pri.append(P_pri)

            ########
            # update
            ########

            # AEIMMKF
            try:

                x_post, P_post, Lambda = self.update(z[:,self.model_index,:1],x_pri,P_pri)

            # IMM
            except:

                x_post, P_post, Lambda = self.update(z,x_pri,P_pri)
    
            self.x_post.append(x_post)
            self.P_post.append(P_post)
            self.Lambda.append(Lambda)

        # compute mixed a priori estimate and covaraince
        # this does not affect the iteration here and only used for post-processing
        x_pri_out, P_pri_out = self.combined_estimate_and_covariance(self.x_pri,self.P_pri) 

        ##################################################################
        # Estimation with Applications to Tracking and Navigation, pg. 456
        # Step 4 - Mode Probability Update
        ##################################################################
        self.mode_probability_update()

        ##################################################################        
        # Estimation with Applications to Tracking and Navigation, pg. 457
        # Step 5 - Estimate and covariance combination
        ##################################################################
        
        x_post_out, P_post_out = self.combined_estimate_and_covariance(self.x_post,self.P_post)

        return [ self.x_post, self.P_post, # input to next iteration of IMM
                 x_pri_out, x_post_out, P_pri_out, P_post_out, # returned as final output
                 tf.add(self.kf_ctr,1), self.mu]
    
    def predict(self,x,P,model):
        
        assert x is not None
        assert P is not None
        assert model is not None

        self.F, self.Q = model

        x_pri, P_pri = super().predict(x,P)
        
        return x_pri, P_pri

    def set_R(self):

        # indexed R
        try:

            R = self.R[self.kf_ctr,self.model_index]

        except:

            R = self.R

        return R

    def update(self,z,x,P):

        x_post, P_post, Lambda = super().update(z,x,P)
        
        return x_post, P_post, Lambda  

    def compute_mixing_probabilities(self):

        """
        Estimation with Applications to Tracking and Navigation, pg. 455
        Step 1 - Calculation of Mixing Probabilities
        """

        # comput c_bar
        self.cbar = self.p.T @ self.mu
        
        # compute mixing parameters
        for i in range(self.n_models):

            for j in range(self.n_models):
                
                self.mix_prob[i][j] = (self.p[i][j] * self.mu[i]) / self.cbar[j]
            
    def compute_mixed_state_and_covariance(self,x,P):

        """
        Estimation with Applications to Tracking and Navigation, pg. 455-456
        Step 2 - Mixing
        """

        # mixed initial conditions and covariance lists
        x_post_star = list()
        P_post_star = list()

        # compute mixed a priori estimate
        for j in range(self.n_models):

            # temporary sum variable
            mixed_x = np.zeros(self.x0.shape).astype(np_float_prec)

            for i in range(self.n_models):

                mixed_x = tf.add(mixed_x,tf.multiply(x[i],self.mix_prob[i][j]))

            x_post_star.append(mixed_x)

        # compute mixed a priori estimate covariance
        for j in range(self.n_models):

            # temporary sum variable
            mixed_P = np.zeros(self.P0.shape).astype(np_float_prec)

            for i in range(self.n_models):

                # difference between i-th model's estimate and j-th mixed initial condition
                x_diff = x[i] - x_post_star[j]

                mixed_P = tf.add(mixed_P,self.mix_prob[i][j] * (P[i] + tf.matmul(x_diff,x_diff,transpose_b=True)))

            P_post_star.append(mixed_P)

        return x_post_star, P_post_star
            
    def mode_probability_update(self):

        """
        Estimation with Applications to Tracking and Navigation, pg. 456
        Step 4 - Mode Probability Update
        """

        self.mu = tf.multiply(tf.squeeze(self.cbar),tf.stack(self.Lambda))
        self.mu /= tf.reduce_sum(self.mu)
        self.mu = tf.expand_dims(self.mu,axis=-1)

    def combined_estimate_and_covariance(self,x,P):

        """
        Estimation with Applications to Tracking and Navigation, pg. 457
        Step 5 - Estimate and covariance combination
        """

        x_post = np.zeros(self.x0.shape).astype(np_float_prec)
        P_post = np.zeros(self.P0.shape).astype(np_float_prec)
        
        # final state estimate
        for j in range(self.n_models):

            x_post = tf.add(x_post,tf.multiply(x[j],self.mu[j]))
        
        # final state estimate covaraince
        for j in range(self.n_models):

            x_diff = x - x_post
            P_post_j = self.mu[j] * (P[j] + tf.matmul(x_post,x_post,transpose_b=True))
            P_post = tf.add(P_post,P_post_j)
            
        return x_post, P_post

################################################################################

class InteractingMultipleModelExtendedKalmanFilter(InteractingMultipleModelKalmanFilter):

    """
    Tensorflow implementation of IMM with Extended Kalman Filter
    """

    def __init__(self,
                 meas_dims=None,
                 state_dims=None,
                 dt=None,
                 model_order=None,
                 models=None,
                 H=None,
                 R=None,
                 x0=None,
                 P0=None,
                 p=None,
                 mu=None):

        super().__init__(meas_dims=meas_dims,
                         state_dims=state_dims,
                         dt=dt,
                         model_order=model_order,
                         models=models,
                         H=H,
                         R=R,
                         x0=x0,
                         P0=P0,
                         p=p,
                         mu=mu)
        
    def predict(self,x,P,model):
        
        assert x is not None
        assert P is not None
        assert model is not None

        self.F, F_params, self.Q = model
        
        # State transition and Jacobian parameters
        self.F_params = {k:v for k,v in self.__dict__.items() if k in F_params}

        x_pri = tf.matmul(self.F(**self.F_params,x=x),x,name='x_pri')

        J = tf.squeeze(jacobian(x_pri,x))

        P_pri = tf.add(J@P@tf.transpose(J),self.Q,name='P_pri')
        
        return x_pri, P_pri
