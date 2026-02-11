#Task State Feedback Control Law
#The overall code is from TADA (e.g., a_forward.m in TADA) 
#and Saltzman & Munhall (1989). Here, we compute xdotdot 
#which is equivalent to tvdotdot in TADA.

#Nam, H., Goldstein, L., Saltzman, E., & Byrd, D. (2004). 
#TADA: An enhanced, portable Task Dynamics model in MATLAB. 
#The Journal of the Acoustical Society of America, 115(5), 2430-2430.

# Saltzman, E. L., & Munhall, K. G. (1989). 
# A dynamical approach to gestural patterning 
# in speech production. Ecological psychology, 
#1(4), 333-382.


import numpy as np
import math
import copy
import global_variables as gv
import pdb
from .util import string2dtype_array

class TaskSFCLaw():
    def run(self,x_tilde,TV_SCORE,i_frm):
        d_BLEND = np.zeros(len(TV_SCORE))
        x_0 = np.zeros(len(TV_SCORE))
        k_BLEND = np.zeros(len(TV_SCORE))
        PROMACT= np.zeros(len(TV_SCORE))
    
        MAXPROM = 1
    
        for i_TV in range(len(TV_SCORE)):
            PROMACT[i_TV] = min(MAXPROM, math.ceil(copy.deepcopy(TV_SCORE[i_TV][0]["PROMSUM"][i_frm])))
            x_0[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["xBLEND"][i_frm])
            k_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["kBLEND"][i_frm])
            d_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["dBLEND"][i_frm])
        
        #print(x_tilde_record)        
        B_times_xdot_tilde =  np.dot(np.diag(d_BLEND),x_tilde[gv.x_dim:2*gv.x_dim])
        K_times_x_tildeminusx0 = np.dot(np.diag(k_BLEND),x_tilde[0:gv.x_dim] - x_0) # using the old one without the 'spi' pi gesture functionality and etc
        xdotdot= -B_times_xdot_tilde -K_times_x_tildeminusx0 # task state acceleration (xdotdot)
        
        return xdotdot, PROMACT

class TaskSFCLaw_inertia_SBI():
    def __init__(self,TaskSFCLaw_configs):
        #pdb.set_trace()
        self.inertial_estimate = float(TaskSFCLaw_configs['inertial_estimate'])

    def run(self,x_tilde,TV_SCORE,i_frm):
        d_BLEND = np.zeros(len(TV_SCORE))
        x_0 = np.zeros(len(TV_SCORE))
        k_BLEND = np.zeros(len(TV_SCORE))
        PROMACT= np.zeros(len(TV_SCORE))
    
        MAXPROM = 1
    
        for i_TV in range(len(TV_SCORE)):
            PROMACT[i_TV] = min(MAXPROM, math.ceil(copy.deepcopy(TV_SCORE[i_TV][0]["PROMSUM"][i_frm])))
            x_0[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["xBLEND"][i_frm])
            k_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["kBLEND"][i_frm])
            d_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["dBLEND"][i_frm])
        
        #print(x_tilde_record)        
        B_times_xdot_tilde =  np.dot(np.diag(d_BLEND),x_tilde[gv.x_dim:2*gv.x_dim])
        K_times_x_tildeminusx0 = np.dot(np.diag(k_BLEND),x_tilde[0:gv.x_dim] - x_0) # using the old one without the 'spi' pi gesture functionality and etc
        xdotdot= (-B_times_xdot_tilde -K_times_x_tildeminusx0) /  self.inertial_estimate# task state acceleration (xdotdot)
        
        return xdotdot, PROMACT

class TaskSFCLaw_with_noise():
    def __init__(self,TaskSFCLaw_configs):
        #pdb.set_trace()
        self.noise_scale = float(TaskSFCLaw_configs['noise_scale'])
        self.xdotdot_mean = string2dtype_array(TaskSFCLaw_configs['xdotdot_mean'], dtype='float') # get an empirical values for this

    def run(self,x_tilde,TV_SCORE,i_frm):
        d_BLEND = np.zeros(len(TV_SCORE))
        x_0 = np.zeros(len(TV_SCORE))
        k_BLEND = np.zeros(len(TV_SCORE))
        PROMACT= np.zeros(len(TV_SCORE))
    
        MAXPROM = 1
    
        for i_TV in range(len(TV_SCORE)):
            PROMACT[i_TV] = min(MAXPROM, math.ceil(copy.deepcopy(TV_SCORE[i_TV][0]["PROMSUM"][i_frm])))
            x_0[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["xBLEND"][i_frm])
            k_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["kBLEND"][i_frm])
            d_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["dBLEND"][i_frm])
        
        #print(x_tilde_record)        
        B_times_xdot_tilde =  np.dot(np.diag(d_BLEND),x_tilde[gv.x_dim:2*gv.x_dim])
        K_times_x_tildeminusx0 = np.dot(np.diag(k_BLEND),x_tilde[0:gv.x_dim] - x_0) # using the old one without the 'spi' pi gesture functionality and etc
        xdotdot= -B_times_xdot_tilde -K_times_x_tildeminusx0 # task state acceleration (xdotdot)

        #pdb.set_trace()
        noise_scale = self.noise_scale
        xdotdot_norm = self.xdotdot_mean
        # pdb.set_trace()
        x_dotdot_noise = 1e0 * noise_scale * np.random.normal(0,1,xdotdot.shape) * xdotdot_norm # add median maybe
        x_dotdot_plus_noise = xdotdot.copy()
        x_dotdot_plus_noise += x_dotdot_noise
        
        return x_dotdot_plus_noise, PROMACT