import numpy as np
from fractions import Fraction as frac
import warnings as Wrn

from tools import *

def A_vector(w, A, tau, Tot, t):
   return A *step(t)*np.sin(w*t) #(np.sin(np.pi * (t - tau) / Tot))**2 * np.cos(w * t) * step(Tot - (t - tau)) * step(t - tau)

def e_k(t_0, vec_k, delta):
    """Calculates the kinetic energy term for the Hamiltonian."""
    ek = 0 + 0j
    for d in delta:
        dot_product = np.dot(vec_k, d).item()  # Convert to scalar
        ek += np.exp(1.0j * dot_product)
    return -t_0 * ek

def V_T_spc(n, m, params, vec_k, t):
    """Calculates the T_spc-dependent perturbation."""
    #sx = np.array([[0, 1], [1, 0]])
    #return (5.0+2.3j)*sx
    w = params[7]
    Vt = (Tot_Ham(params, vec_k, t)-Ham(params, vec_k))*(np.exp(1j * w * (n - m) * t))
    return Vt
    #return A * (np.sin(np.pi * (t - tau) / Tot))**2 * np.cos(w * t) * step(Tot - (t - tau)) * step(t - tau) * sx

def Ham(params, vec_k):
    """Constructs the static Hamiltonian for given parameters and momentum."""
    hdim, E_g, t_0, delta = params[0], params[3], params[4], params[5]
    H = np.zeros((hdim, hdim), dtype=np.cdouble)

    ek = e_k(t_0, vec_k, delta)
    for i in range(hdim):
      for j in range(hdim):
        if(i==j):
          H[i, j] = (-1)**i * E_g / 2.0
        elif i < j:
          H[i, j] = ek
        else:
          H[i, j] = np.conjugate(H[j, i])

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Unperturbed Hamiltonian is not Hermitian!")

    return H

def Tot_Ham(params, vec_k, t,tau=0, qe=-1):
    """Constructs the full T_spc dependent Hamiltonian for given parameters and momentum."""
    hdim, E_g, t_0, delta, A, w, Tot = params[0], params[3], params[4], params[5], params[6], params[7], params[8]
    H = np.zeros((hdim, hdim), dtype=np.cdouble)
    for i in range(hdim):
      for j in range(hdim):
        pierls_k = vec_k + qe*A_vector(w, A, tau, Tot, t)*(i-j)
        if(i==j):
          H[i, j] = (-1)**i * E_g / 2.0
        elif i < j:
          H[i, j] = e_k(t_0, pierls_k, delta)
        else:
          H[i, j] = np.conjugate(H[j, i])

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Unperturbed Hamiltonian is not Hermitian!")

    return H

def Floq_Ham(F_modes,V_T_spc, params,T_spc, vec_k):
  hdim, w, Tot, Nm = params[0], params[7], params[8], params[11]
  H_F = np.zeros((Nm,Nm,hdim, hdim), dtype=np.cdouble)
  H_F_comb = np.zeros((Nm*hdim, Nm*hdim), dtype=np.cdouble)
  H = Ham(params, vec_k)
  #Vt = V_T_spc(w, A, 0.0, Tot, 0.0)
  #Euler Integral
  '''Vt = simpson_integrate(V_T_spc, (n, m, params, vec_k), T_spc, rule="euler")/ Tot'''
  #Simpson's 1/3 integral
  '''Vt = simpson_integrate(V_T_spc, (n, m, params, vec_k), T_spc, rule="1/3") / Tot'''
  #Simpson's 3/8 integral
  for i in range(Nm):
    n = F_modes[i]
    for j in range(Nm):
      m = F_modes[j]
      Vt = simpson_integrate(V_T_spc, (n, m, params, vec_k), T_spc, rule="3/8")/ Tot
      if(n==m):
        H_F[i,j,:,:]=H-(n*w)*np.eye(hdim)
      if(abs(n-m)==1):
        if n < m:
            H_F[i,j,:,:] = Vt
        else:
            H_F[i,j,:,:] = np.conjugate(H_F[j, i,:,:])

  H_F_comb = H_F.transpose(0, 2, 1, 3).reshape(Nm*hdim, Nm*hdim)
  if not np.allclose(H_F_comb, H_F_comb.conj().T):
        raise ValueError("The Floquet Hamiltonian is not Hermitian!")
  return H_F_comb

def Parallel_Ham(F_modes,V_T_spc, params,T_spc, k_space):
  hdim, Nk, Nm = params[0], params[9], params[11]
  H_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  E_F = np.zeros((Nk,Nm*hdim), dtype=float)
  V_F = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
  for kp,vec_k in enumerate(k_space):
    H_F[kp,:,:] = Floq_Ham(F_modes,V_T_spc, params,T_spc, vec_k)
    E_F[kp,:], V_F[kp,:,:] = np.linalg.eigh(H_F[kp,:,:])
    #formatted_eigenvalues = " | ".join(f"{x:.3f}" for x in E_F[kp, :])
    #print(f"kp={kp} and vec_k={vec_k}\nEigenvalues:\n{formatted_eigenvalues}")
    #print('Hamiltonian:')
    #print_formatted_matrix(H_F[kp,:,:])
    #print('Eigenvectors:')
    #print_formatted_matrix(V_F[kp,:,:])
    #V_F_flat[kp,:,:,:,:] = V_F[kp,:,:].reshape(Nm,hdim, Nm,hdim).transpose(0, 2, 1, 3)
  return E_F, V_F

def eig_vec(params, F_modes, E_F, V_F, t):
    hdim, w, Nk, Nm = params[0], params[7], params[9], params[11]
    
    # Initialize arrays
    nexp = np.zeros((Nm * hdim, hdim), dtype=np.cdouble)
    chi_mode = np.zeros((Nk, Nm * hdim, Nm, hdim), dtype=np.cdouble)
    chi = np.zeros((Nk, Nm * hdim, hdim), dtype=np.cdouble)
    chi_final = np.zeros((Nk, hdim, hdim), dtype=np.cdouble)
    E_final = np.zeros((Nk, hdim), dtype=float)
    I = np.zeros((Nm * hdim, Nm*hdim), dtype=np.cdouble)
    
    # Compute chi
    orthnorm = []
    for kp in range(Nk):
        # Populate chi_mode based on V_F
        #print(f"{kp=}")
        for alpha in range(Nm * hdim):
            for i in range(hdim):
                S = 0
                for beta in range(Nm):
                    n = F_modes[beta]
                    chi_mode[kp, alpha, beta, i] = V_F[kp,beta * hdim + i, alpha]
                    #print(f"VF[{beta * hdim + i}, {alpha}]=chi^({alpha})_{beta} [{i}]")
                    S += np.exp(1j * n * w * t) * chi_mode[kp, alpha, beta, i]
                nexp[alpha,i] = S
                chi[kp, alpha,i] = np.exp(-1j * E_F[kp, alpha] * t) * nexp[alpha,i]
        for i in range(Nm*hdim):
            for j in range(Nm*hdim):
                I[i,j] = np.vdot(chi[kp,i,:],chi[kp,j,:])
                if(np.allclose(I[i,j],0.0+0.0j)):
                   #print(f"{i} and {j} are orthogonal!")
                   cond1 = E_F[kp,i] < w/2
                   cond2 = E_F[kp,i] >= -w/2
                   if(kp==0 and cond1 and cond2):
                      orthnorm.append(i)
        if(len(orthnorm)<hdim):
           Wrn.warn(f"The number of orthogonal vectors found is less than {hdim}")
        elif(len(orthnorm)>hdim):
           Wrn.warn(f"The number of orthogonal vectors found is more than {hdim}") 
        for k, o in enumerate(orthnorm[:hdim]): #for k,o in enumerate(orthnorm):
           #print(f"{k=} and {o=}")
           chi_final[kp,k,:] = chi[kp,o,:]
           E_final[kp,k] = E_F[kp,o]
        #print_formatted_matrix(I)
    #if not np.allclose(I[i,j], np.eye(Nm*hdim)):
    #    raise ValueError("The chi wavevectors are not orthogonal!")
    return E_final, chi_final, chi_mode, orthnorm

def TD_wfn(params,F_modes,EF,VF,V,k_space,Tspc):
    hdim, Nk, Nt, Nm = params[0], params[9], params[10], params[11]
    X = np.zeros((Nk,Nm * hdim, hdim), dtype=np.cdouble)
    w = np.zeros((Nk,hdim), dtype=np.cdouble)
    psi = np.zeros((Nt, Nk, hdim), dtype=np.cdouble)
    for tp, t in enumerate(Tspc):
        E_final, chi_final, chi_mode, orthnorm = eig_vec(params, F_modes, EF, VF,t)
        for kp, vec_k in enumerate(k_space):
            #print(f"{tp=},{kp=}")
            S3 = np.zeros_like(np.outer(X[0, 0, :], X[0, 0, :]))
            for alpha in orthnorm:
                S = np.zeros_like(chi_mode[0, 0, 0, :])
                for n in range(Nm):
                    S = S + chi_mode[kp, alpha, n, :]
                X[kp, alpha, :] = S
                S3 = S3 + np.outer(X[kp, alpha, :], X[kp, alpha, :].conj())
            S3 = S3/S3[0,0]
            #print_formatted_matrix(S3)
            if not np.allclose(S3, np.eye(hdim)):
               raise ValueError("The condition XX^t is not satisfied")
            else:
               S = np.zeros_like(chi_mode[0, 0, 0, :])
               for k,alpha in enumerate(orthnorm):
                  w[kp,k]=np.vdot(X[kp, alpha, :]/S3[0,0],V[kp,:,k])
                  #print(f"alpha:w[{kp},{k}]={w[kp,k]}")
                  S = S + w[kp,k]*chi_final[kp,k,:]
               psi[tp,kp,:] = S
               psi[tp,kp,:] = psi[tp,kp,:]/np.linalg.norm(psi[tp,kp,:])
    return E_final,psi