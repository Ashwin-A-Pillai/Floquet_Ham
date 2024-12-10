import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import FuncFormatter, MultipleLocator

from Ham import *
from tools import *
from current import *

# Parameters and initialization
sdim = 1
hdim = 2
a_cc = 1.0
t_0 = 1.0
E_g = 0.0
Nk = 101
Nt = 100
mod_lim = 2
Nm = 2*mod_lim+1
w = 0.25 #np.pi
A = 0.5 #100
Tot = (2 * np.pi) / w
my_dpi = 700

T_spc = np.linspace(0, 1.5 * Tot, Nt)
F_modes = np.arange(-mod_lim, mod_lim + 1)

delta = np.array([[a_cc], [-a_cc]])
         #  0    1     2     3    4     5    6  7   8   9   10  11    12
params = [hdim, sdim, a_cc, E_g, t_0, delta, A, w, Tot, Nk, Nt, Nm, mod_lim]
k_space = np.linspace(-np.pi / 2.0, np.pi / 2.0, Nk)

color_map = ['red', 'blue', 'violet', 'cyan', 'pink','royalblue', 'maroon', 'teal', 'salmon', 'deepskyblue']
color_map_pos = ['blue', 'cyan', 'royalblue', 'teal', 'deepskyblue']
color_map_neg = ['red', 'violet', 'pink', 'maroon', 'salmon']

H = np.zeros((Nk, hdim, hdim), dtype=np.cdouble)
E = np.zeros((Nk, hdim), dtype=float)
V = np.zeros((Nk, hdim, hdim), dtype=np.cdouble)

VF = np.zeros((Nk,Nm*hdim,Nm*hdim), dtype=np.cdouble)
EF = np.zeros((Nk,Nm*hdim), dtype=float)

chi=np.zeros((Nk, hdim, hdim), dtype=np.cdouble)
EF_final = np.zeros((Nk, hdim), dtype=float)
I_H = np.zeros(Nm,dtype=np.cdouble)

psi = np.zeros((Nt, Nk, hdim), dtype=np.cdouble)

I = np.zeros(Nt, dtype=np.cdouble)
I_mod = np.zeros((Nm,hdim,hdim),dtype=np.cdouble)

for kp,vec_k in enumerate(k_space):
  H[kp,:,:] = Ham(params, vec_k)
  E[kp,:], V[kp,:,:] = np.linalg.eigh(H[kp,:,:])
  for a in range(hdim):
    plt.scatter(vec_k,E[kp,a], color=color_map[a])
    V[kp,:,a] = V[kp,:,a]/np.linalg.norm(V[kp,:,a])
plt.savefig('1D-BS.png',dpi=my_dpi, bbox_inches='tight')
plt.close()
#H_F=Floq_Ham(F_modes,params, vec_k)

EF,VF = Parallel_Ham(F_modes,V_T_spc, params,T_spc, k_space)
for h in range(Nm*hdim):
  plt.plot(k_space,EF[:,h],color=color_map[h])
  VF[kp,:,h] = VF[kp,:,h]/np.linalg.norm(VF[kp,:,h])
plt.xlabel('k')
plt.ylabel('E(k)')
plt.savefig('Floquet Eigenbands.png', dpi=my_dpi, bbox_inches='tight')
plt.close()
#for t in T_spc:
#    EF_final, chi, chi_mod, Ort = eig_vec(params, F_modes, EF, VF,t)
E_final,psi,chi_mode, w_k, ON_al = TD_wfn(params,F_modes,EF,VF,V,k_space,T_spc)
#E_final, psi = TD_wfn(params,F_modes,EF,VF,V,k_space,T_spc)

I_H = HHC(params,chi_mode, w_k, ON_al, F_modes,k_space,T_spc)
plt.plot(F_modes, I_H.real, color='blue', label='Real part')
plt.plot(F_modes, I_H.imag, color='orange', label='Imaginary part')
plt.xlabel('Time')
plt.ylabel(r'$I_H$')
plt.legend()
plt.title('HHC vs Time')
plt.savefig('HHC.png', dpi=my_dpi, bbox_inches='tight')
plt.close()
#H_F=Floq_Ham(F_modes,params, vec_k)
#I_mod = cur_mode(F_modes, params,vec_k, T_spc)
I = total_current(params,psi,k_space,T_spc)

plt.plot(T_spc, I.real, color='blue', label='Real part')
plt.plot(T_spc, I.imag, color='orange', label='Imaginary part')
plt.xlabel('Time')
plt.ylabel('Current')
plt.legend()
plt.title('Current vs Time')
plt.savefig('Total current.png', dpi=my_dpi, bbox_inches='tight')
plt.close()
# Fourier Transform
freq = np.fft.fftfreq(len(T_spc), d=(T_spc[1] - T_spc[0]))  # Frequencies
fft_result = np.fft.fft(I)

# Magnitude of the FFT (for peak detection)
fft_magnitude = np.abs(fft_result)

# Find peaks in the FFT magnitude
peaks, properties = find_peaks(fft_magnitude, height=0.1)  # Adjust height as needed

# Plot the FFT
plt.figure(figsize=(10, 6))
plt.plot(freq, fft_magnitude, label='FFT Magnitude')
plt.scatter(freq[peaks], fft_magnitude[peaks], color='red', label='Peaks')
plt.xlabel('Frequency (1/T_spc units)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of I')
plt.legend()
plt.grid()
plt.savefig('FFT_Peaks.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the frequencies of the peaks
print("Peak Frequencies:", freq[peaks])
'''
W_set = [1.0,0.25]
F_set = [0, 0.5]
Char = ['a','b','d','e']
y_lim = [1,0.5,1.0,0.125]
#x_lim1 = [0,0,0,np.pi/3.0]
x_lim2 = [np.pi/3.0, np.pi/3.0,np.pi/2.0,np.pi/2.0]
k_space = np.linspace(0, np.pi, Nk)
E_g = 0.2
ind = 0
for w in W_set:
  for F in F_set:
    A = F*w

    params = [hdim, sdim, a_cc, E_g, t_0, delta, A, w, Tot, Nk, Nt, Nm, mod_lim]
    
    EF,VF = Parallel_Ham(F_modes,V_T_spc, params,T_spc, k_space)
    #EF,VF = Parallel_Ham(F_modes,params, k_space)

    for a in range(Nm*hdim):
        plt.plot(k_space,EF[:,a], label=f'Column-{a}')
    plt.ylim(-y_lim[ind],y_lim[ind])
    plt.xlim(0,x_lim2[ind])

    # Set x-axis tick locations and labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi / 6))  # Ticks at intervals of pi/2
    ax.xaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))  # Format ticks as fractions of pi

    plt.legend()
    plt.title(f"For F = {F} and $\\Omega$ = {w}")
    plt.savefig(f'1D-BS-Floquet_{Char[ind]}.png',dpi=my_dpi, bbox_inches='tight')
    plt.close()
    ind+=1'''