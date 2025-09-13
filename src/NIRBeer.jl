module NIRBeer

using PCHIPInterpolation, DataFrames

export pchip_interpolation, outlier_detection

include(joinpath(@__DIR__, "helper.jl"))

function pchip_interpolation(df::DataFrame, batch::String, type::Symbol)
    # Validate the type argument
    if type ∉ [:ethanol, :extract]
        throw(ArgumentError("type must be either :ethanol or :extract, got :$type"))
    end
    
    # Filter for the specific batch
    df_batch = filter(row -> row.BatchName == batch, df)
    
    if isempty(df_batch)
        throw(ArgumentError("Batch '$batch' not found in the DataFrame"))
    end
    
    # Extract data
    cumulative_time = convert(Vector{Float64}, df_batch.CumulativeTime)
    
    # Perform PCHIP interpolation based on the requested type
    if type == :extract
        return PCHIPInterpolation.Interpolator(cumulative_time, df_batch.Ereal_mean)
    else 
        return PCHIPInterpolation.Interpolator(cumulative_time, df_batch.wtPercEtOH_mean)
    end
end

struct PCAModel
    μ::Vector{Float64} # column means
    P::Matrix{Float64} # eigenvectors (p x n_components)
    λ::Vector{Float64} # eigenvalues (n_components)
    λ_all::Vector{Float64} # all eigenvalues
end
 

using LinearAlgebra, Statistics


function fit_pca(X::AbstractMatrix, var_target::Float64=0.95, max_components::Int=50)

    n, p = size(X)
    @assert n > max_components "Number of samples must be greater than max_components"
    @assert p > max_components "Number of features must be greater than max_components"
    @assert 0 < var_target < 1 "var_target must be in (0,1)"

    μ = mean(X, dims=1)
    X_centered = X .- μ

    # Calculate covariance matrix
    cov_matrix = cov(X_centered)

    # Eigen decomposition
    eigenvalues, eigenvectors = eigen(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = sortperm(eigenvalues, rev=true)
    λ_sorted = eigenvalues[sorted_indices]
    v_sorted = eigenvectors[:, sorted_indices]
    
    return PCAModel(vec(μ), v_sorted[:, 1:max_components], λ_sorted[1:max_components], eigenvalues)
end

function pca_scores(X::AbstractMatrix, m::PCAModel)

    Z = X .- m.μ'

    T = Z * m.P

    return T

end

function pca_reconstruct(T::AbstractMatrix, m::PCAModel)

    Xhat_centered = T * m.P'

    return Xhat_centered

end


function hotellings_T2(T::AbstractMatrix, λ::AbstractVector)

    invλ = 1.0 ./ λ
    n = size(T, 1)
    T2 = similar(T, n)

    @inbounds for i in 1:n
        s = 0.0
        for j in 1:length(λ)
            s += (T[i, j]^2) * invλ[j]
        end
        T2[i] = s
    end
    return T2
end

function q_spe(X::AbstractMatrix, m::PCAModel)

    Z = X .- m.μ'
    T = Z * m.P
    R = Z .- T * m.P'
    n = size(X, 1)

    Q = similar(T, n)
    @inbounds for i in 1:n
        Q[i] = sum(abs2, @view R[i, :])
    end

    return Q

end

using Distributions

function t2_limit(alpha::Float64, k::Int, n::Int)

    Fcrit = quantile(FDist(k, n - k), alpha)
    return (k * (n - 1) / (n - k)) * Fcrit

end


# Jackson–Mudholkar Q-limit using eigenvalues of residual PCs
function q_limit(alpha::Float64, λ_all::AbstractVector, k::Int)

    # residual eigenvalues: indices k+1:end

    if k >= length(λ_all)

        return Inf

    end

    λr = @view λ_all[(k+1):end]
    θ1 = sum(λr)
    θ2 = sum(λr .^ 2)
    θ3 = sum(λr .^ 3)

    h0 = 1.0 - (2.0 * θ1 * θ3) / (3.0 * θ2^2)

    z = quantile(Normal(), alpha)
    term = z * sqrt(2.0 * θ2 * h0^2) / θ1 + 1.0 + (θ2 * h0 * (h0 - 1.0)) / (θ1^2)

    return θ1 * (term^(1.0 / h0))

end

function detect_outliers_pca(X::AbstractMatrix, alpha::Float64, var_target::Float64, max_components::Int)

    n, _ = size(X)

    model = fit_pca(X, var_target, max_components)

    T = pca_scores(X, model)

    T2 = hotellings_T2(T, model.λ)

    Q  = q_spe(X, model)

    k = length(model.λ)

    T2_lim = t2_limit(alpha, k, n)

    Q_lim  = q_limit(alpha, model.λ_all, k)

    keep = map(eachindex(T2)) do i

        (T2[i] ≤ T2_lim) && (Q[i] ≤ Q_lim)

    end

    return keep, model, T2, Q, T2_lim, Q_lim

end


function iterative_clean_pca(X::AbstractMatrix, alpha::Float64, var_target::Float64, max_components::Int, 
    max_iter::Int, max_drop_frac::Float64, stop_tol::Int)

    n = size(X, 1)
    idx = collect(1:n)
    history = Vector{NamedTuple{(:iter,:n,:k,:n_dropped,:T2_lim,:Q_lim)}}()
    last_dropped = 0

    for it in 1:max_iter
        Xsub = X[idx, :]
        keep, model, T2, Q, T2_lim, Q_lim = detect_outliers_pca(Xsub, alpha, var_target, max_components)
        idx_keep = idx[keep]
        n_drop = length(idx) - length(idx_keep)
        push!(history, (iter=it, n=length(idx), k=length(model.λ), n_dropped=n_drop, T2_lim=T2_lim, Q_lim=Q_lim))

        # Safety checks
        if n_drop == 0 || abs(n_drop - last_dropped) ≤ stop_tol
            idx = idx_keep
            break
        end

        if n_drop / n > max_drop_frac
            # too aggressive — relax alpha or var_target, or break
            println("Reaching max_drop_frac at iteration $it, stopping.")
            idx = idx_keep
            break
        end
        last_dropped = n_drop
        idx = idx_keep
    end

    return idx, history

end


function outlier_detection(X::AbstractMatrix; alpha::Float64=0.05, var_target::Float64=0.95, max_components::Int=50,
    max_iter::Int=5, max_drop_frac::Float64=0.1, stop_tol::Int=5)

    
     return iterative_clean_pca(X, alpha, var_target, max_components, max_iter, max_drop_frac, stop_tol)
end
end 