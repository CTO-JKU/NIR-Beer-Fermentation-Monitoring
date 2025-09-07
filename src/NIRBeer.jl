module NIRBeer

using PCHIPInterpolation, DataFrames

export pchip_interpolation

include(joinpath(@__DIR__, "helper.jl"))

using PCHIPInterpolation

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

end