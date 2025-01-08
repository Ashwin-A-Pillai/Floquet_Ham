using LinearAlgebra
using SpecialFunctions  # For Bessel functions
using PyPlot


# Constants (assuming these are defined elsewhere)
Nm = 1  # Example value, replace with actual value
Q = 0.1  # Example value, replace with actual value
W = 1.0  # Example value, replace with actual value
F = 0.5  # Example value, replace with actual value
Nk = 100 # Example value, replace with actual value

# Pauli matrices
σ0 = [1 0; 0 1]
σ1 = [0 1; 1 0]
σ3 = [1 0; 0 -1]

# Define the Hamiltonian matrix H[k]
function H(k)
    H_matrix = zeros(ComplexF64, (2*(2*Nm+1), 2*(2*Nm+1)))
    for n in -Nm:Nm
#        for m in -Nm:Nm
        for m in n:Nm
            if n == m
                H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] = (n * W * σ0 + Q * σ3)
            end
            H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2] += (1.0im)^(m-n) * besselj(m-n, F) * cos(k - (m-n)*π/2) * σ1
            H_matrix[2*(m+Nm)+1:2*(m+Nm)+2, 2*(n+Nm)+1:2*(n+Nm)+2] = conj(H_matrix[2*(n+Nm)+1:2*(n+Nm)+2, 2*(m+Nm)+1:2*(m+Nm)+2])
        end
    end
    return H_matrix
end

# Define a0[k]
a0(k) = -1/sqrt(2) * sqrt(1 - Q / sqrt(Q^2 + cos(k)^2))

# Define b0[k]
b0(k) = 1/sqrt(2) * sqrt(1 + Q / sqrt(Q^2 + cos(k)^2))

# Define E0[k]
E0(k) = -sqrt(Q^2 + cos(k)^2)

# Define A[k] (eigenvector corresponding to the (2*Nm+1)-th eigenvalue)
function A(k)
    eig = eigen(H(k))
#    sorted_indices = sortperm(eig.values)
#    return eig.vectors[:, sorted_indices[2*Nm+1]]
    return eig.vectors[:, 2*Nm+1]
end

# Define Xa[k]
function Xa(k)
    A_vec = A(k)
    return [sum(A_vec[2i-1] for i in 1:2*Nm+1), sum(A_vec[2i] for i in 1:2*Nm+1)]
end

# Define wa[k]
wa(k) = conj(Xa(k)) ⋅ [a0(k), b0(k)]

# Define B[k] (eigenvector corresponding to the (2*Nm+2)-th eigenvalue)
function B(k)
    eig = eigen(H(k))
#    sorted_indices = sortperm(eig.values)
#    return eig.vectors[:, sorted_indices[2*Nm+2]]
    return eig.vectors[:, 2*Nm+2]
end

# Define Xb[k]
function Xb(k)
    B_vec = B(k)
    return [sum(B_vec[2i-1] for i in 1:2*Nm+1), sum(B_vec[2i] for i in 1:2*Nm+1)]
end

# Define wb[k]
wb(k) = conj(Xb(k)) ⋅ [a0(k), b0(k)]


kpts=range(0,pi/2.0,Nk)
band_struct=zeros(ComplexF64,Nk, (2*(2*Nm+1)))

for (ik,k) in enumerate(kpts)
    print("Doing $k is H(k) hermitian $(ishermitian(H(k))) \n")
    diag_H = eigen(H(k))
    band_struct[ik,:]=diag_H.values
end

for n in 1:2*(2*Nm+1)
  plot(kpts,band_struct[:,n], label="Band $n")
end
PyPlot.show()

wa2=abs.(wa.(kpts).^2)
wb2=abs.(wb.(kpts).^2)

plot(kpts,wb2-wa2, label="W difference")
PyPlot.show()
exit(0)
# Define Bound[l]
Bound(l) = (-Nm-1 < l < Nm+1) ? 1 : 0

# Define ChiA[k, j]
function ChiA(k, j)
    if -Nm-1 < j < Nm+1
        return [A(k)[2*(j+Nm)+1], A(k)[2*(j+Nm)+2]]
    else
        return [0, 0]
    end
end

# Define ChiB[k, j]
function ChiB(k, j)
    if -Nm-1 < j < Nm+1
        return [B(k)[2*(j+Nm)+1], B(k)[2*(j+Nm)+2]]
    else
        return [0, 0]
    end
end

# Define IHknl[k, N0, n, l]
function IHknl(k, N0, n, l)
    term1 = abs(wa(k))^2 * conj(ChiA(k, n-l+N0)) ⋅ (σ1 * ChiA(k, n))
    term2 = abs(wb(k))^2 * conj(ChiB(k, n-l+N0)) ⋅ (σ1 * ChiB(k, n))
    return (term1 + term2) * im^l * besselj(l, F) * sin(k + l*π/2)
end

# Define IHk[N0, k]
function IHk(N0, k)
    return sum(IHknl(k, N0, n, l) for n in -Nm:Nm, l in -Nm:Nm)
end

# Define IH[N0]
function IH(N0)
    return sum(IHk(N0, -π/2 + π/Nk * i) for i in 0:Nk)
end
