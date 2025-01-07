#
# Simple Tight-binding code for one-dimensional system
# Claudio Attaccalite (2024)
#
using Printf
using LinearAlgebra
using PyPlot
using Bessels
using DelimitedFiles
using ArgParse
include("tb_parms.jl")
using .TB_parms

# Define Pauli matrices
σ₀ = I(2)  # Identity matrix
σ₁ = [0 1; 1 0]
σ₃ = [1 0; 0 -1]


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
        "-w"
            help = "write on disk results"
            action  = :store_true
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
	H=Q*σ₃ 
        # Off diagonal part
        #
        H+=σ₁*cos(k[1]+At)
	return H
end

function Floquet_Hamiltonian(k, F_modes;  Q=0.0, omega=1.0, F=0.0, damp=0.0)
        h_size =2
        n_modes=length(F_modes)
        H_flq=zeros(Complex{Float64},n_modes,n_modes,h_size,h_size)

#Diagonal terms respect to the mode and Q
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          H_flq[i1,i1,:,:]=i_m*omega*σ₀+Q*σ₃-1im*damp*σ₀
        end
        
#Off-diagonal terms in mode 
        for i1 in 1:n_modes
          i_m=F_modes[i1]
          for i2 in i1:n_modes
             i_n=F_modes[i2]
             H_flq[i1,i2,:,:]+=(1.0im)^(i_m-i_n)*besselj(i_m-i_n,F)*cos(k[1]-(i_m-i_n)*pi/2.0)*σ₁ 
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
	kpath=Any[]
	for i in 1:(n_points-1)
		for j in 0:(n_steps-1)	
			dp=(points[i+1]-points[i])/n_steps
			push!(kpath,points[i]+dp*j)
		end
	end
	push!(kpath,points[n_points])
	return kpath
end


function main()
    parsed_args = parse_commandline()
    if parsed_args["w"]
        TB_parms.write_on_disk=true
    end
    #
    println("\n\n * * * TB-Floquet code * * * \n\n")
    #
    # generate k-points list
    # 
    n_kpt=parsed_args["nkpt"] 
    zero=[0.0]
    Pi2=[+pi/(2.0)]
    kpoints=[zero,Pi2]
    kpath=generate_circuit(kpoints,n_kpt)
    #
    # Build distanced between k-points
    #
    nkpt=length(kpath)
    kdist=zeros(Float64,nkpt)
    kpt1=kpath[1]
    for (ik,kpt) in enumerate(kpath)
        dk=kpt-kpt1
        kdist[ik]=sum(abs,dk)
        if ik>1
            kdist[ik]+=kdist[ik-1]
        end
        kpt1=kpt
    end
    #    
    #tb-parameters
    #
    Q=parsed_args["Q"]
    if parsed_args["t"]
        band_structure,eigenvecs=TB_diag(kpath,Q)
        plot(kdist,band_structure[:, 1], label="Band 1")
        plot(kdist,band_structure[:, 2], label="Band 2")
        title("Band structure for two site 1D model")
        PyPlot.show()
        if(TB_parms.write_on_disk)
          tb_out=open("TB_Hamiltonian.txt", "w")
          write(tb_out,"# Q = $Q \n#\n")
          write(tb_out,"# kpt   E[1]    E[2]\n")
          writedlm(tb_out, [kdist band_structure[:,1] band_structure[:,2]],"     ")
          close(tb_out)
        end
    end
    #
    #
    # Floquet Hamiltonian parameters
    #
    F       =parsed_args["F"]    # Intensity
    omega   =parsed_args["omega"] # Frequency
    max_mode=parsed_args["nmax"]  # max number of modes
    damp    =parsed_args["damp"]  # max number of modes
    n_modes =max_mode*2+1
    imod=round(Int,(n_modes-1)/2+1)
    if parsed_args["f"]
        flq_bands,flq_eigenvecs,F_modes=FLQ_diag(kpath,Q,omega,F,max_mode,damp)
        plot(kdist,flq_bands[:, 1, imod], label="Mode 1 band 1")
        plot(kdist,flq_bands[:, 2, imod], label="Mode 1 band 2")
#        for n in 1:n_modes
#          plot(kdist,flq_bands[:, n,1], label="Mode $n band 1")
#          plot(kdist,flq_bands[:, n,2], label="Mode $n band 2")
#        end
        title("Floquet band structure for two site 1D model")
        PyPlot.show()
        if(TB_parms.write_on_disk)
          for n in 1:n_modes
              imod=F_modes[n]
              flq_out=open("FLQ_Hamiltonian_mode_$imod.txt", "w")
              write(flq_out,"# Q = $Q \n#\n")
              write(flq_out,"# Mode = $imod \n#\n")
              write(flq_out,"# kpt   E[1]    E[2]\n")
              writedlm(flq_out, [kdist flq_bands[:,n,1] flq_bands[:,n,2]],"     ")
             close(flq_out)
           end
        end
        #
        # Calculate xhi_alpha 
        xhi_alpha=Build_xhi_alpha(nkpt,n_modes,flq_eigenvecs)
        #
        # Generate initial wave-function psi_0 = eigenvec[1,:]
        #
        band_structure,eigenvecs=TB_diag(kpath, Q)
        psi_0=eigenvecs[:,1,:]
        #
        # Calculate weights in the Floquet basis and check normalizations
        weights  =Build_weights(nkpt,psi_0,xhi_alpha,n_modes)
        weights_diff=abs.(weights[:,2].^2)-abs.(weights[:,1].^2)
        plot(kdist, weights_diff, label="Weights_diff")
        title("Weights diff")
        PyPlot.show()
        #
    end
    #
    # Real-time
    #
    if parsed_args["r"]
       # RT_dynamics()
    end
end

function TB_diag(kpath,Q)
  println("")
  println(" * * * Tight-binding code for 1D-system  * * *")
  println("")

  band_structure = zeros(Float64, length(kpath), 2)
  eigenvecs      = zeros(Complex{Float64}, length(kpath), 2, 2)
  
  for (ik,kpt) in enumerate(kpath)
  	H=Hamiltonian(kpt,Q)
  	diag_H = eigen(H)                  # Diagonalize the matrix
        band_structure[ik, :] = diag_H.values  # Store eigenvalues in an array
        eigenvecs[ik,:, : ]    = diag_H.vectors
  end
  return band_structure,eigenvecs
end


function FLQ_diag(kpath,Q,omega,F,max_mode,damp)
  h_size=2    # Hamiltonian size
  #
  F_modes=range(-max_mode,max_mode,step=1)
  n_modes=length(F_modes)
  #
  println("")
  @printf("Floquet Hamiltonian Q=%f  F=%f  omega=%f max_mode=%d ",Q,F,omega,max_mode)
  println("")
  #
  nkpt=length(kpath)
  flq_bands    = zeros(Float64, nkpt, h_size, n_modes)
  flq_eigenvec = zeros(Complex{Float64}, nkpt, h_size*n_modes, h_size, n_modes)
  all_eigenvec = zeros(Complex{Float64}, nkpt, n_modes*h_size, n_modes*h_size)
  #
  for (ik,kpt) in enumerate(kpath)
  	H_flq=Floquet_Hamiltonian(kpt,F_modes;Q,omega,F,damp)
        diag_H_flq=eigen(H_flq)
  	eigenvalues  = diag_H_flq.values       # Diagonalize the matrix
        eigenvectors = diag_H_flq.vectors
        flq_bands[ik, :,:]        = reshape(eigenvalues,(h_size, n_modes))  # Store eigenvalues in an array
        all_eigenvec[ik, :,:]     = eigenvectors
        flq_eigenvec[ik,:,:,:]  = reshape(eigenvectors,(h_size*n_modes, h_size, n_modes))  # Store eigenvalues in an array
  end
  return flq_bands,flq_eigenvec,F_modes
 end
  #
function Build_xhi_alpha(nkpt,n_modes,flq_eigenvecs)
  h_size=2
  xhi_alpha = zeros(Complex{Float64}, nkpt, n_modes*h_size, h_size)
  println("Build X_alpha ..")
    # Check ortormality
  #
  println("Check normalization xhi_alpha ..")
  M=zeros(Complex{Float64},h_size,h_size)
  for ik in 1:nkpt
      dot11=dot(flq_eigenvecs[ik,1,:,:],flq_eigenvecs[ik,1,:,:])
      dot12=dot(flq_eigenvecs[ik,3,:,:],flq_eigenvecs[ik,2,:,:])
      print(" Dot $dot11  and    $dot12   \n")
  end
  imod=round(Int,(n_modes-1)/2+1)
  for ik in 1:nkpt
    for n in 1:n_modes
      xhi_alpha[ik,:,1]+=flq_eigenvecs[ik,:,1,n]
      xhi_alpha[ik,:,2]+=flq_eigenvecs[ik,:,2,n]
    end
  end
  #
  # Renormalization Xhi_alpha
  #
#  xhi_alpha.=xhi_alpha/sqrt(n_modes)
  #
  # Check ortormality
  #
  println("Check normalization xhi_alpha ..")
  for ik in 1:nkpt
    dot11=dot(xhi_alpha[ik,:,1],xhi_alpha[ik,:,1])
    dot12=dot(xhi_alpha[ik,:,2],xhi_alpha[ik,:,1])
     print(" Dot $dot11  and    $dot12   \n")
  end 
  #
  exit(0)
  return xhi_alpha
 end

 function Build_weights(nkpt,psi_0,xhi_alpha,n_modes)
  #
  h_size=2
  println("Build weights ..")
  weights = zeros(Complex{Float64}, nkpt, h_size*n_modes)
  for ik in 1:nkpt
    for n in 1:(h_size*n_modes)
      weights[ik,n]=dot(xhi_alpha[ik,:,n],psi_0[ik,:])  # xhi^+ \dot \psi_0
    end
  end
  #
  # Check normalization
  #
  println("Check normalization ..")
  for ik in 1:nkpt
      if !isapprox(norm(weights[ik,:]),1.0; atol=1e-7)
          print("Error in normalization for ik= $ik  \n ")
      end
  end 
  return weights
 end
  #
  function calculate_current()
  # Calculate current
  #
   n_max=n_modes
   #
   I_hN=zeros(Complex{Float64},n_max)
   println("Build current coefficent ..")
   for iN in 1:n_max
     for (ik,kpt) in enumerate(kpath)
     for ia in 1:h_size
        I_aKN=Build_I_alpha_kN(ik,kpt,iN,ia,flq_eigenvec,n_max,F,imode)
        print("ik  $ik  I_akN: ",abs.(I_aKN))
        I_hN[iN]+=(weights[ik,ia,:]'weights[ik,ia,:])*I_aKN[ia]
    
     end
     end
   end
   print("I_hN coefficents : ",abs.(I_hN))

end

function Build_I_alpha_kN(ik,k,iN,ia,flq_eigenvec,n_max,F,imode)
   h_size=2
   I_alpha_kN=zeros(Complex{Float64},h_size)
   for l in 1:n_max
      I_kl=Build_I_kN(k,l,F)
      for n in 1:n_max
         inp=n-l+iN
         if inp <1 || inp >n_max
             continue
         end
         I_alpha_kN[ia]+=flq_eigenvec[ik,imode,ia,inp,:]'*I_kl*flq_eigenvec[ik,imode,ia,iN,:]
      end
   end
 return I_alpha_kN
end

function Build_I_kN(k,n,F)
  h_size=2
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
