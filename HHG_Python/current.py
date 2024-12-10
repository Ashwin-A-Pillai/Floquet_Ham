import numpy as np

from tools import *
from Ham import *

def cur(params, vec_k ,t):
  hdim, A, w,Tot,Nt = params[0], params[6], params[7], params[8], params[10]
  dt = 1/Nt
  I = np.zeros((hdim,hdim),dtype=np.cdouble)
  H_p = Tot_Ham(params, vec_k, t+dt)
  H_m = Tot_Ham(params, vec_k, t)
  A_p = A_vector(w, A, 0, Tot, t+dt)
  A_m = A_vector(w, A, 0, Tot, t)
  num = (H_p-H_m)/dt
  den = (A_p-A_m)/dt
  I[:,:] = num/den
  return I

def cur_mode(F_modes, params,vec_k, T_spc):
  hdim, Nm = params[0], params[11]
  I_mod = np.zeros((Nm,hdim,hdim),dtype=np.cdouble)
  def core_I(n,params,vec_k,t):
    w = params[7]
    return cur(params,vec_k,t)*np.exp(1j*n*w*t)
  for i,n in enumerate(F_modes):
    I_mod[i,:,:] = simpson_integrate(core_I, (n, params, vec_k), T_spc, rule="3/8")
    print(f"{vec_k=} and {n=}")
    print_formatted_matrix(I_mod[i,:,:])
  return I_mod

def total_current(params,psi,k_space,T_spc):
   Nt = params[10]
   I_full = np.zeros(Nt,dtype=np.cdouble)
   for tp, t in enumerate(T_spc):
    LI = 0.0 +0.0j
    for kp,vec_k in enumerate(k_space):
       RI = cur(params, vec_k ,t)@psi[tp,kp,:]
       LI = LI + np.vdot(psi[tp,kp,:],RI)
    I_full[tp] = LI
   return I_full