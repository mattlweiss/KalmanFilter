import tensorflow as tf
import numpy as np

def F_CT(dt,x):

    w = x[4,0]
    wt = w*dt

    # if w is zero
    f0 = np.array([
        [1.0,dt,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0,0.0],
        [0.0,0.0,1.0,dt,0.0],
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,0.0,1.0]
    ])

    # if w is non-zero
    f11 = [1.0,tf.sin(wt)/w,0.0,-(1-tf.cos(wt))/w,0.0]
    f12 = [0.0,tf.cos(wt),0.0,-tf.sin(wt),0.0]
    f13 = [0.0,(1-tf.cos(wt))/w,1.0,tf.sin(wt)/w,0.0]
    f14 = [0.0,tf.sin(wt),0.0,tf.cos(wt),0.0]
    f15 = [0.0,0.0,0.0,0.0,1.0]
    f1 = tf.stack([f11,f12,f13,f14,f15])

    #return tf.cond(tf.less(tf.abs(w),1e-2),lambda:f0,lambda:f1)
    return f1
    
def J_CT_tf(dt,x):

    x_dot,y_dot = x[1,0], x[3,0]

    w = x[4,0]
    wt = w*dt

    # if w is zero
    j01 = [1.0,dt,0.0,0.0,-0.5*(dt**2)*y_dot]
    j02 = [0.0,1.0,0.0,0.0,-dt*y_dot]
    j03 = [0.0,0.0,1.0,dt,0.5*(dt**2)*x_dot]
    j04 = [0.0,0.0,0.0,1.0,dt*x_dot]
    j05 = [0.0,0.0,0.0,0.0,1.0]
    j0 = tf.stack([j01,j02,j03,j04,j05])

    # if w is non-zero
    from tensorflow.python.ops.parallel_for.gradients import jacobian

    f1 = [1.0,tf.sin(wt)/w,0.0,(tf.cos(wt)-1.0)/w,0.0]
    f2 = [0.0,tf.cos(wt),0.0,-tf.sin(wt),0.0]
    f3 = [0.0,(1-tf.cos(wt))/w,1.0,tf.sin(wt)/w,0.0]
    f4 = [0.0,tf.sin(wt),0.0,tf.cos(wt),0.0]
    f5 = [0.0,0.0,0.0,0.0,1.0]
    f = tf.stack([f1,f2,f3,f4,f5])

    j1 = jacobian(f@x,x)
    j1 = tf.squeeze(j1)
    J = tf.cond(tf.less(tf.abs(w),1e-2),lambda:j0,lambda:j1)
    
    return J

def J_CT_manual(dt,x):

    x, y, x_dot,y_dot, w = x[0,0], x[1,0], x[2,0], x[3,0], x[4,0]
    wt = w*dt

    # if w is zero
    j01 = [1.0,dt,0.0,0.0,-0.5*(dt**2)*y_dot]
    j02 = [0.0,1.0,0.0,0.0,-dt*y_dot]
    j03 = [0.0,0.0,1.0,dt,0.5*(dt**2)*x_dot]
    j04 = [0.0,0.0,0.0,1.0,dt*x_dot]
    j05 = [0.0,0.0,0.0,0.0,1.0]
    j0 = tf.stack([j01,j02,j03,j04,j05])

    # if w is non-zero
    j11 = [1.0, tf.sin(wt)/w,     0.0, -(1-tf.cos(wt))/w,  x_dot*( (tf.cos(wt)*dt)/w - tf.sin(wt)/w**2) - y_dot*( (tf.sin(wt)*dt)/w + (-1+tf.cos(wt))/w**2)]
    j12 = [0.0, tf.cos(wt),       0.0, -tf.sin(wt),        -x_dot*tf.sin(wt)*dt - y_dot*tf.cos(wt)*dt]
    j13 = [0.0, (1-tf.cos(wt))/w, 1.0,  tf.sin(wt)/w,      x_dot*( (tf.sin(wt)*dt)/w - (1-tf.cos(wt))/w**2) + y_dot*( (tf.cos(wt)*dt)/w - (tf.sin(wt)*dt)/w**2)]
    j14 = [0.0, tf.sin(wt),       0.0,  tf.cos(wt),        x_dot*tf.cos(wt)*dt - y_dot*tf.sin(wt)*dt]
    j15 = [0.0, 0.0,              0.0, 0.0,                1.0]
    j1 = tf.stack([j11,j12,j13,j14,j15])
    
    #return tf.cond(tf.equal(w,0.0),lambda:j_eps,lambda:j)
    return tf.cond(tf.less(tf.abs(w),1e-2),lambda:j0,lambda:j1)
