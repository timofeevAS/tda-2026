using CSV
using DataFrames
using Statistics
using DelayEmbeddings

const DATA_PATH = "dataset_clean.csv"
const TAU_RESULTS_PATH = "uzal_embedding_results_all_columns.csv"
const OUTPUT_PATH = "fnn_embedding_results_all_columns.csv"

const TIME_COL = "time"

const MIN_N = 20
const MIN_DIM = 3
const MAX_DIM = 12

const FNN_THRESHOLD = 0.05
const IFNN_R = 2
const THEILER_WINDOW = 1


function safe_float_vector(x)
    result = Float64[]

    for value in x
        if ismissing(value)
            continue
        end

        try
            push!(result, Float64(value))
        catch
            continue
        end
    end

    return result
end


function get_tau_for_column(tau_df::DataFrame, colname::String)
    idx = findfirst(==(colname), tau_df.timeseries_name)

    if idx === nothing
        return missing
    end

    τ = tau_df[idx, :selected_tau]

    if ismissing(τ)
        return missing
    end

    return Int(τ)
end


function choose_fnn_dim(
    x::Vector{Float64},
    τ::Int;
    min_dim::Int=MIN_DIM,
    max_dim::Int=MAX_DIM,
    fnn_threshold::Float64=FNN_THRESHOLD
)
    n = length(x)

    if n < MIN_N || τ < 1
        return missing
    end

    possible_max_dim = min(max_dim, max(2, (n - 1) ÷ τ))

    if possible_max_dim < min_dim
        return missing
    end

    dims = collect(min_dim:possible_max_dim)

    fnn_values = delay_ifnn(
        x,
        τ,
        dims;
        r=IFNN_R,
        w=THEILER_WINDOW
    )

    fnn_fractions = Float64.(fnn_values)

    for (i, frac) in enumerate(fnn_fractions)
        if isfinite(frac) && frac <= fnn_threshold
            return dims[i]
        end
    end

    finite_idx = findall(isfinite, fnn_fractions)

    if isempty(finite_idx)
        return missing
    end

    best_i = finite_idx[argmin(fnn_fractions[finite_idx])]

    return dims[best_i]
end


function main()
    df = CSV.read(DATA_PATH, DataFrame)
    tau_df = CSV.read(TAU_RESULTS_PATH, DataFrame)

    results = DataFrame(
        timeseries_name = String[],
        n = Int[],
        tau = Union{Missing, Int}[],
        fnn_dim = Union{Missing, Int}[]
    )

    for col in names(df)
        if col == TIME_COL
            continue
        end

        name = string(col)
        x = safe_float_vector(df[:, col])
        n = length(x)

        τ = get_tau_for_column(tau_df, name)

        if n < MIN_N || ismissing(τ)
            push!(results, (name, n, τ, missing))
            continue
        end

        dim = choose_fnn_dim(x, τ)

        push!(results, (name, n, τ, dim))
    end

    CSV.write(OUTPUT_PATH, results)

    println(results)
    println("Saved to: ", OUTPUT_PATH)
end


main()