import sys                     
import tensorflow as tf
tf_float_prec = tf.float64
import numpy as np
np_float_prec = np.float64
import tensorflow_probability as tfp
from sklearn.datasets import make_spd_matrix
from scipy import stats
from pdb import set_trace as st
from dovebirdia.utilities.base import saveDict
from dovebirdia.math.linalg import is_invertible, pos_diag

from tensorflow.python.ops.parallel_for.gradients import jacobian

class KalmanFilter():

    def __init__(self,
                 meas_dims=None,
                 state_dims=None,
                 dt=None,
                 model_order=None,
                 F=None,
                 Q=None,
                 H=None,
                 R=None,
                 x0=None,
                 P0=None):

        """
        Implements a Kalman Filter in Tensorflow
        """

        self.meas_dims = meas_dims
        self.state_dims = state_dims
        self.model_order = model_order
        self.dt = dt
        
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x0 = x0
        self.P0 = P0

        self.sample_freq = np.reciprocal(self.dt)

    def fit(self,inputs):

        """
        Apply Kalman Filter, Using Wrapper Functions
        inputs is a list.  First element is z, second (optional) element is R
        """

        z = self.process_inputs(inputs)

        x_hat_pri, x_hat_post, P_pri, P_post, self.kf_ctr = tf.scan(self.kfScan,
                                                                    z,
                                                                    initializer = [ self.x0,
                                                                                    self.x0,
                                                                                    self.P0,
                                                                                    self.P0,
                                                                                    tf.constant(0) ],
                                                                    name='kfScan')
        
        filter_results = self.process_results(x_hat_pri, x_hat_post, P_pri, P_post, z)
        
        return filter_results

    def kfScan(self,state, z):

        """ This is where the Kalman Filter is implemented. """

        _, x_post, _, P_post, self.kf_ctr = state

        ##########
        # Predict
        ##########
        
        x_pri, P_pri = self.predict(x_post,P_post)

        #########
        # Update
        #########

        x_post, P_post, _ = self.update(z, x_pri, P_pri)

        return [ x_pri, x_post, P_pri, P_post, tf.add(self.kf_ctr,1) ]

    def predict(self,x=None,P=None):

        assert x is not None
        assert P is not None

        x_pri = tf.matmul(self.F,x,name='x_pri')
        P_pri = tf.add(self.F@P@tf.transpose(self.F),self.Q,name='P_pri')
        
        return x_pri, P_pri

    def set_R(self):

        # indexed R
        try:

            R = self.R[self.kf_ctr,0]

        # Fixed R
        except:

            R = self.R

        return R
    
    def update(self,z,x,P):

        assert z is not None
        assert x is not None
        assert P is not None

        R = self.set_R()

        S = self.H@P@tf.transpose(self.H) + R 
        S_inv = tf.linalg.inv(S)

        K = tf.matmul(P,tf.matmul(self.H,S_inv,transpose_a=True,name='KF_H-S_inv'),name='KF_K')

        y = tf.subtract(z,tf.matmul(self.H,x),name='innov_plus')

        x_post = tf.add(x,tf.matmul(K,y),name='x_post')
        P_post = (tf.eye(tf.shape(P)[0],dtype=tf_float_prec)-K@self.H)@P

        # compute likelihood
        likelihood = self.MultivariateNormalLikelihood(y,S)
        likelihood += tf.cast(1e-8,dtype=tf_float_prec)
        
        return x_post, P_post, likelihood

    def process_inputs(self,inputs):

        # if learning R, z and R will be passed
        # assume z and R are tensors since inputs is a list only for the AEKF
        if isinstance(inputs,list):

            # extract z and (possibly) R from inputs list
            z = inputs[0]
            z = tf.convert_to_tensor(z)

            self.R = inputs[1]

            # ensure z is rank 3
            tf.cond(tf.rank(3)<3,lambda:tf.expand_dims(z,axis=-1),lambda:z)

        # fixed R or sample R
        else:

            # if R is not passed set z
            # ensure z is rank 3
            if np.ndim(inputs) < 3:

                z = np.expand_dims(inputs,axis=-1)
                
            # if self.R is non use sample covariance
            if self.R is None:

                z_hat = np.squeeze(z) - np.mean(np.squeeze(z),axis=0)
                self.R = z_hat.T@z_hat / (z_hat.shape[0]-1)
                
        return tf.convert_to_tensor(z)

    def process_results(self,x_hat_pri, x_hat_post, P_pri, P_post, z, mu=None):

        z_hat_pri  = tf.matmul(self.H, x_hat_pri, name='z_pri', transpose_b=False)
        z_hat_post = tf.matmul(self.H, x_hat_post, name='z_post', transpose_b=False)
        HPHT_pri = self.H@P_pri@tf.transpose(self.H)
        HPHT_post = self.H@P_post@tf.transpose(self.H)

        if mu is None:

            mu = z
        
        filter_result = {
            'x_hat_pri':x_hat_pri,
            'x_hat_post':x_hat_post,
            'z_hat_pri':z_hat_pri,
            'z_hat_post':z_hat_post,
            'P_pri':P_pri,
            'P_post':P_post,
            'HPHT_pri':HPHT_pri,
            'HPHT_post':HPHT_post,
            'z':z,
            'R':tf.convert_to_tensor(self.R),
            'mu':mu
            }

        return filter_result

    def evaluate(self,x=None, x_key='z_hat_post', save_results=True):

        assert x is not None

        filter_result = self.fit(x)

        return filter_result[x_key][:,:,0], filter_result['R']

    def MultivariateNormalLikelihood(self,y,S):
                   
        k = tf.cast(y.get_shape()[0],dtype=tf_float_prec)
        a = tf.transpose(y)@tf.linalg.inv(S)@(y)
        Z = (2*np.pi)**(-k/2.0) * tf.linalg.det(S)**(-0.5)

        return (Z * tf.exp(-a/2.0))[0][0]
    
class ExtendedKalmanFilter(KalmanFilter):

    """
    Tensorflow implementation of Extended Kalman Filter
    """

    def __init__(self,
                 meas_dims=None,
                 state_dims=None,
                 dt=None,
                 model_order=None,
                 F=None,
                 F_params=None,
                 Q=None,
                 G=None,
                 H=None,
                 R=None,
                 x0=None,
                 P0=None):

        super().__init__(meas_dims=meas_dims,
                         state_dims=state_dims,
                         dt=dt,
                         model_order=model_order,
                         F=F,
                         Q=Q,
                         H=H,
                         R=R,
                         x0=x0,
                         P0=P0)

        # State transition and Jacobian parameters
        self.F_params = {k:v for k,v in self.__dict__.items() if k in F_params}

    def predict(self,x=None,P=None):

        assert x is not None
        assert P is not None

        F = self.F(**self.F_params,x=x)
        x_pri = tf.matmul(F,x,name='x_pri')
        
        J = tf.squeeze(jacobian(x_pri,x))

        P_pri = tf.add(J@P@tf.transpose(J),self.Q,name='P_pri')
        
        return x_pri, P_pri
