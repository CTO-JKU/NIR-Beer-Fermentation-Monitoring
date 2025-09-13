# Trim first/last noisy wavenumbers (adjust n_edge to your instrument)
function trim_edges(X::AbstractMatrix; n_edge::Int=30)

    p = size(X, 2)

    @assert 2n_edge < p "n_edge too large for number of columns"

    return @view X[:, (n_edge+1):(p-n_edge)]

end

# Standard Normal Variate per spectrum
function snv!(X::AbstractMatrix)
    n, p = size(X)
    @inbounds for i in 1:n
        μ = mean(@view X[i, :])
        σ = std(@view X[i, :])
        σ > 0 ? (X[i, :] .-= μ; X[i, :] ./= σ) : (X[i, :] .-= μ)  # guard zero-variance
    end
    return X
end

# Multiplicative Scatter Correction (needs reference spectrum, e.g., mean)
function msc!(X::AbstractMatrix; ref::AbstractVector=nothing)
    n, p = size(X)
    ref === nothing && (ref = vec(mean(X, dims=1)))
    # Fit y ≈ a*ref + b per spectrum and correct: (y - b)/a
    @inbounds for i in 1:n
        y = @view X[i, :]
        A = hcat(ref, ones(p))
        # least squares [a,b]
        ab = A \ y
        a, b = ab[1], ab[2]
        X[i, :] .= (y .- b) ./ a
    end
    return X
end