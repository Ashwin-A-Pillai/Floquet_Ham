#
# Simple Tight-binding code for one-dimensional system
# Claudio Attaccalite (2024)
#
using Printf
using LinearAlgebra
using PyPlot
using Bessels
using ArgParse
include("tb_parms.jl")
using .TB_parms


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-t"
            help = "diagonalize tb-Hamiltonian"
            action = :store_true
        "-f"
            help = "diagonalize flq-Hamiltonian"
            action = :store_true
        "-r"
            help = "real-time dynamics"
            action = :store_true
        "--Q"
            help = "gap parameter"
            arg_type = Float64
            default  = TB_parms.Q
        "--nkpt"
            help = "number of k-ponints"
            arg_type = Int
            default  = TB_parms.nkpt
        "--nmax"
            help = "max number of modes"
            arg_type = Int
            default  = TB_parms.max_mode
        "--F"
            help = "Field intensity"
            arg_type = Float64
            default  = TB_parms.F
        "--omega"
            help = "Field frequency"
            arg_type = Float64
            default  = TB_parms.omega
        "--tstep"
            help = "Time-step for real-time dynamics"
            arg_type = Float64
            default  = TB_parms.tstep
        "--nsteps"
            help = "Number of steps for real-time dynamics"
            arg_type = Int
            default  = TB_parms.n_steps
        "--damp"
            help = "Damping parameter"
            arg_type = Float64
            default  = TB_parms.damp
    end

    return parse_args(s)
end

# In this program the lattice constant is equal to 1

function Hamiltonian(k,Q;At=0.0)::Matrix{Complex{Float64}}
        #
	H=zeros(Complex{Float64},2,2)
        #
        # Diagonal part 0,E_gap
        #
	H[1,1]=-Q
	H[2,2]=Q
        #
        # Off diagonal part
        #
        H[1,2]=cos(k[1]+At)
	H[2,1]=conj(H[1,2])
	return H
end

function Floquet_Hamiltonian(k, F_modes;  Q=0.0, omega=1.0, F=0.0, damp=0.0)
        h_size =2
        n_modes=length(F_modes)
        H_flq=zeros(Complex{Float64},n_modes,n_modes,h_size,h_size)

#Diagonal terms respect to the mode and Q
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for ih in (1:h_size)
            H_flq[i1,i1,ih,ih]=i_m*omega+Q*(-1.0)^ih-1im*damp
          end
        end
        
#Off-diagonal terms in mode 
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for i2 in i1:n_modes
             i_n=F_modes[i2]
             H_flq[i1,i2,1,2]=(1.0im)^(i_m-i_n)*besselj(i_m-i_n,F)*cos(k[1]-(i_m-i_n)*pi/2.0)
             H_flq[i1,i2,2,1]=H_flq[i1,i2,1,2]
             if i1 != i2
                 H_flq[i2,i1,:,:]=conj(H_flq[i1,i2,:,:])
             end
          end
       end
 #My own reshape
       return copy(reshape(permutedims(H_flq,(1,3,2,4)),(n_modes*h_size,n_modes*h_size)))
end


    
#
function generate_circuit(points, n_steps)
	println("Generate k-path ")
	n_points=length(points)
	@printf("number of points:%5i \n",n_points)
	if n_points <= 1
		error("number of points less or equal to 1 ")
	end
	for i in 1:n_points
		@printf("v(%d) = %s \n",i,points[i])
	end
	path=Any[]
	for i in 1:(n_points-1)
		for j in 0:(n_steps-1)	
			dp=(points[i+1]-points[i])/n_steps
			push!(path,points[i]+dp*j)
		end
	end
	push!(path,points[n_points])
	return path
end


function main()
    parsed_args = parse_commandline()
    #
    println("\n\n * * * TB-Floquet code * * * \n\n")
    #
    # generate k-points list
    # 
    n_kpt=parsed_args["nkpt"] 
    zero=[0.0]
    Pi2=[+pi/(2.0)]
    kpoints=[zero,Pi2]
    path=generate_circuit(kpoints,n_kpt)
    #    
    #tb-parameters
    #
    Q=parsed_args["Q"]
    if parsed_args["t"]
        band_structure,eigenvecs=TB_diag(path,Q)
        plot(band_structure[:, 1], label="Band 1")
        plot(band_structure[:, 2], label="Band 2")
        title("Band structure for two site 1D model")
        PyPlot.show()
    end
    #
    #
    # Floquet Hamiltonian parameters
    #
    F       =parsed_args["F"]    # Intensity
    omega   =parsed_args["omega"] # Frequency
    max_mode=parsed_args["nmax"]  # max number of modes
    damp    =parsed_args["damp"]  # max number of modes
    if parsed_args["f"]
        FLQ_diag(path,Q,omega,F,max_mode,damp)
    end
    #
    # Real-time
    #
    if parsed_args["r"]
       # RT_dynamics()
    end
end

function TB_diag(path,Q)
  println("")
  println(" * * * Tight-binding code for 1D-system  * * *")
  println("")

  band_structure = zeros(Float64, length(path), 2)
  eigenvecs      = zeros(Complex{Float64}, length(path), 2, 2)
  
  for (ik,kpt) in enumerate(path)
  	H=Hamiltonian(kpt,Q)
  	diag_H = eigen(H)                  # Diagonalize the matrix
        band_structure[ik, :] = diag_H.values  # Store eigenvalues in an array
        eigenvecs[ik,:, : ]    = diag_H.vectors
  end
  return band_structure,eigenvecs
end


function FLQ_diag(path,Q,omega,F,max_mode,damp)
  h_size=2    # Hamiltonian size
  #
  F_modes=range(-max_mode,max_mode,step=1)
  n_modes=length(F_modes)
  #
  band_structure,eigenvecs=TB_diag(path, Q)
  #
  println("")
  @printf("Floquet Hamiltonian Q=%f  F=%f  omega=%f max_mode=%d ",Q,F,omega,max_mode)
  println("")
  #
  nkpt=length(path)
  flq_bands    = zeros(Float64, nkpt, n_modes, h_size)
  flq_eigenvec = zeros(Complex{Float64}, nkpt, n_modes, h_size, n_modes, h_size)
  all_eigenvec = zeros(Complex{Float64}, nkpt, n_modes*h_size, n_modes*h_size)
  #
  for (ik,kpt) in enumerate(path)
  	H_flq=Floquet_Hamiltonian(kpt,F_modes;Q,omega,F,damp)
        diag_H_flq=eigen(H_flq)
  	eigenvalues  = diag_H_flq.values       # Diagonalize the matrix
        eigenvectors = diag_H_flq.vectors
        flq_bands[ik, :,:]        = reshape(eigenvalues,(n_modes,h_size))  # Store eigenvalues in an array
        all_eigenvec[ik, :,:]     = eigenvectors
        flq_eigenvec[ik,:,:,:,:]  = reshape(eigenvectors,(n_modes,h_size,n_modes,h_size))  # Store eigenvalues in an array
  end
  plot(flq_bands[:, 1,1], label="Mode 1 band 1")
  plot(flq_bands[:, 1,2], label="Mode 1 band 2")
  for n in 2:n_modes
    plot(flq_bands[:, n,1], label="Mode $n band 1")
    plot(flq_bands[:, n,2], label="Mode $n band 2")
  end
  title("Floquet band structure for two site 1D model")
  PyPlot.show()
  #
  # Psi_0=eigenvec[ik,1,:]
  #
  # Build \chi^\alpha
  #
  xhi_alpha = zeros(Complex{Float64}, nkpt, h_size, h_size)
  imode=max_mode+1  # I choose the eigenvector with mode=0
  println("Build X_alpha ..")
  for ik in 1:nkpt
    for n in 1:n_modes
      xhi_alpha[ik,1,:]+=flq_eigenvec[ik,imode,1,n,:]
      xhi_alpha[ik,2,:]+=flq_eigenvec[ik,imode,2,n,:]
    end
  end
  # 
  # Build weights
  #
  println("Build weights ..")
  weights = zeros(Complex{Float64}, nkpt, h_size, h_size)
  for ik in 1:nkpt
      weights[ik,1,:].=conj(xhi_alpha[ik,1,:]).*eigenvecs[ik,1,:]  # xhi^+ \dot \psi_0
      weights[ik,2,:].=conj(xhi_alpha[ik,2,:]).*eigenvecs[ik,1,:]
  end
  #
  # Check normalization
  #
  println("Check normalization ..")
  for ik in 1:nkpt
     print("Norm $ik : ",norm(weights[ik,1,:])^2+norm(weights[ik,2,:])^2," \n ")
  end 
  #
  # Calculate current
  #
  n_max=n_modes
  #
  I_hN=zeros(Complex{Float64},n_max)
  println("Build current coefficent ..")
  Build_I_alpha_kN(flq_eigenvec,n_max,nkpt,F,imode)
  for iN in 1:n_max
    for ik in 1:nkpt, ia in 1:h_size
       I_aKN=Build_I_alpha_kN(ik,iN,ia,flq_eigenvec,n_max,F,imode)
       I_hN+=(weights[ik,ia,:]'weights[ik,ia,:])*I_aKN
    end
  end

end

function Build_I_alpha_kN(ik,iN,ia,flq_eigenvec,n_max,F,imode)
     I_alpha_kN=zeros(Complex{Float64},h_size)
     for l in 1:n_max
        I_kl=Build_I_kN(k,l,F)
        for n in 1:n_max
          inp=n-l+iN
          I_alpha_kN+=flq_eigenvec[ik,imode,ia,inp,:]*I_kl*flq_eigenvec[ik,imode,ia,in,:]
        end
     end
     return I_alpha_kN
end

function Build_I_kN(k,n,F)
  I_kN=zeros(Complex{Float64},h_size,h_size)
  I_kN[1,2]=I_kN[2,1]=(-1im)^n*besselj(n,F)*sin(k[1]+n*pi/2.0)
  return I_kN
end

function RT_dynamics(kpoints,Q,omega,F,tstep,nsteps)
  h_size=2    # Hamiltonian size
  #
  nk=length(kpoints)
  psi0=zeros(Complex{Float64},nk,h_size)
  #
  # Initial WF in the ground-state
  psi0[:,1]=1.0
  #
end

function time_evolution(ψ,time;t0=0.0)
        A_t=0.0
end

main()
