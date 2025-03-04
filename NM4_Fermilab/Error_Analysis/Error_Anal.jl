using Plots
using LinearAlgebra
using Printf
using Statistics
using Measures  

# Constants
const g = 0.05
const s = 0.04
const bigy = sqrt(3 - s)

function GenerateLineshape(P, x)

    function cosal(x, eps)
        return (1 .- eps .* x .- s) ./ bigxsquare(x, eps)
    end

    function bigxsquare(x, eps)
        return sqrt.(g^2 .+ (1 .- eps .* x .- s).^2)
    end

    function mult_term(x, eps)
        return 1.0 ./ (2 .* π .* sqrt.(bigxsquare(x, eps)))
    end

    function cosaltwo(x, eps)
        arg = (1 .+ cosal(x, eps)) ./ 2
        return sqrt.(arg) 
    end

    function sinaltwo(x, eps)
        arg = (1 .- cosal(x, eps)) ./ 2
        return sqrt.(arg) 
    end

    function termone(x, eps)
        return π / 2 .+ atan.((bigy^2 .- bigxsquare(x, eps)) ./ (2 .* bigy .* sqrt.(bigxsquare(x, eps)) .* sinaltwo(x, eps)))
    end

    function termtwo(x, eps)
        return log.((bigy^2 .+ bigxsquare(x, eps) .+ 2 .* bigy .* sqrt.(bigxsquare(x, eps)) .* cosaltwo(x, eps)) ./ 
                (bigy^2 .+ bigxsquare(x, eps) .- 2 .* bigy .* sqrt.(bigxsquare(x, eps)) .* cosaltwo(x, eps)))
    end

    function icurve(x, eps)
        return mult_term(x, eps) .* (2 .* cosaltwo(x, eps) .* termone(x, eps) .+ sinaltwo(x, eps) .* termtwo(x, eps))
    end

    r = (sqrt.(4 .- 3 .* P.^2) .+ P) / (2 .- 2 .* P)
    Iplus = r .* icurve(x, 1) ./ 10
    Iminus = icurve(x, -1) ./ 10
    signal = Iplus .+ Iminus
    return signal, Iplus, Iminus
end

# Parameters
n = 10000 
num_bins = 500  # Number of frequency bins
data_min = -3.0 
data_max = 3.0  
polarization_values = vcat(0.005:0.05:.8) 

# Generate continuous data and bin it for each polarization value
x_values = range(data_min, stop=data_max, length=n)  
bin_edges = range(data_min, stop=data_max, length=num_bins + 1)  # Bin edges
bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2  # Compute bin centers for plotting

binned_errors = zeros(length(polarization_values), num_bins)

# Loop over polarization values
@showprogress for (idx, P) in enumerate(polarization_values)
    println("Processing polarization value $idx of $(length(polarization_values))...")
    
    signal, Iplus, Iminus = GenerateLineshape(P, x_values)
    
    # Bin the data
    bin_indices = searchsortedlast.(Ref(bin_edges), x_values)  # Assign data to bins
    bin_indices = clamp.(bin_indices, 1, num_bins)  # Clip to valid indices
    
    # Compute binned errors
    for i in 1:num_bins
        mask = (bin_indices .== i)
        if any(mask)
            binned_errors[idx, i] = std(signal[mask])  # Standard deviation as error
        end
    end
end

# Normalize the error between 0 and 1
min_error = minimum(binned_errors)
max_error = maximum(binned_errors)
normalized_errors = (binned_errors .- min_error) ./ (max_error - min_error)

println("Creating 3D Surface + 1D Signal Animation...")

forward_indices = 1:length(polarization_values)
backward_indices = (length(polarization_values)-1):-1:1  # Skip the last frame to avoid duplication
animation_indices = vcat(forward_indices, backward_indices)  # Combine forward and backward indices

anim1 = @animate for idx in animation_indices
    P = polarization_values[idx]
    println("Rendering frame $idx of $(length(polarization_values))...")
    
    # 1D Signal Plot
    p1 = plot(x_values, GenerateLineshape(P, x_values)[1], label="Signal (P = $(@sprintf("%.2f", P)))", 
              xlabel="Frequency", ylabel="Signal Intensity", title="1D Signal Plot", 
              linewidth=2, legend=:topright, titlefontsize=12, xguidefontsize=10, yguidefontsize=10)
    
    # 3D Surface Plot
    p2 = surface(bin_centers, polarization_values[1:idx], normalized_errors[1:idx, :], 
                xlabel="Frequency Bin", ylabel="Polarization", zlabel="Normalized Error", 
                title="3D Surface Plot", c=:plasma, legend=false,
                titlefontsize=12, xguidefontsize=10, yguidefontsize=10, zguidefontsize=10)
    
    # Combine plots
    plot(p1, p2, layout=(1, 2), size=(1400, 700), margin=10mm)
end

gif(anim1, "3D_Surface_1D_Signal_Animation.gif", fps=10)

# 2D Heatmap + 1D Signal Plot
println("Creating 2D Heatmap + 1D Signal Animation...")
anim2 = @animate for idx in animation_indices
    P = polarization_values[idx]
    println("Rendering frame $idx of $(length(polarization_values))...")
    
    # 1D Signal Plot
    p1 = plot(x_values, GenerateLineshape(P, x_values)[1], label="Signal (P = $(@sprintf("%.2f", P)))", 
              xlabel="Frequency", ylabel="Signal Intensity", title="1D Signal Plot", 
              linewidth=2, legend=:topright, titlefontsize=12, xguidefontsize=10, yguidefontsize=10)
    
    # 2D Heatmap
    p2 = heatmap(bin_centers, polarization_values[1:idx], normalized_errors[1:idx, :], 
                 xlabel="Frequency Bin", ylabel="Polarization", title="2D Heatmap", c=:inferno, legend=false,
                 titlefontsize=12, xguidefontsize=10, yguidefontsize=10)
                 
    plot(p1, p2, layout=(1, 2), size=(1400, 700), margin=10mm)
end

# Save Animation 2
gif(anim2, "2D_Heatmap_1D_Signal_Animation.gif", fps=10)