using CSV
using DataFrames
using Statistics
using DelayEmbeddings

const DATA_PATH = "dataset_clean.csv"
const UZAL_PATH = "uzal_cost_final_result.csv"
const OUTPUT_PATH = "pecora_res.csv"

const TIME_COL = "time"

const MIN_N = 50
const TMAX = 100

const SAMPLESIZE = 0.3
const K_NEIGHBORS = 13
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


function normalize_series(x::Vector{Float64})
    σ = std(x)

    if σ == 0 || !isfinite(σ)
        return x
    end

    return (x .- mean(x)) ./ σ
end


function first_local_max(v::Vector{Float64}, forbidden::Set{Int})
    if length(v) < 3
        return missing
    end

    for i in 2:length(v)-1
        τ = i - 1

        if τ in forbidden
            continue
        end

        if isfinite(v[i]) && v[i] > v[i - 1] && v[i] > v[i + 1]
            return τ
        end
    end

    return missing
end


function global_max_lag(v::Vector{Float64}, forbidden::Set{Int})
    best_τ = missing
    best_val = -Inf

    for i in eachindex(v)
        τ = i - 1

        if τ in forbidden
            continue
        end

        if isfinite(v[i]) && v[i] > best_val
            best_val = v[i]
            best_τ = τ
        end
    end

    return best_τ
end


function choose_next_lag_by_pecora(
    x::Vector{Float64},
    selected_lags::Vector{Int}
)
    selected_js = ones(Int, length(selected_lags))

    εs, Γs = pecora(
        x,
        Tuple(selected_lags),
        Tuple(selected_js);
        delays = 0:TMAX,
        J = 1:1,
        samplesize = SAMPLESIZE,
        K = K_NEIGHBORS,
        w = THEILER_WINDOW,
        undersampling = false
    )

    ε = vec(εs[:, 1])

    forbidden = Set(selected_lags)

    τ = first_local_max(ε, forbidden)

    if ismissing(τ)
        τ = global_max_lag(ε, forbidden)
    end

    return τ
end


function get_uzal_dim(uzal_df::DataFrame, name::String)
    idx = findfirst(==(name), uzal_df.column)

    if idx === nothing
        return missing
    end

    dim = uzal_df[idx, :dimension]

    if ismissing(dim)
        return missing
    end

    return Int(dim)
end


function pecora_nonuniform_lags(x::Vector{Float64}, target_dim::Int)
    n = length(x)

    if n < MIN_N
        return missing, "", "too_short"
    end

    if target_dim < 1
        return missing, "", "bad_dimension"
    end

    x = normalize_series(x)

    selected_lags = [0]

    while length(selected_lags) < target_dim
        try
            τ = choose_next_lag_by_pecora(x, selected_lags)

            if ismissing(τ)
                return length(selected_lags), join(selected_lags, ";"), "stopped_no_lag"
            end

            push!(selected_lags, Int(τ))

        catch err
            return length(selected_lags), join(selected_lags, ";"), "error: " * string(typeof(err))
        end
    end

    return length(selected_lags), join(selected_lags, ";"), "ok"
end


function main()
    df = CSV.read(DATA_PATH, DataFrame)
    uzal_df = CSV.read(UZAL_PATH, DataFrame)

    results = DataFrame(
        timeseries_name = String[],
        n = Int[],
        target_dim = Union{Missing, Int}[],
        pecora_dim = Union{Missing, Int}[],
        pecora_lags = String[],
        method = String[],
        status = String[]
    )

    for col in names(df)
        if col == TIME_COL
            continue
        end

        name = string(col)
        x = safe_float_vector(df[:, col])
        n = length(x)

        target_dim = get_uzal_dim(uzal_df, name) + 1

        if ismissing(target_dim)
            push!(
                results,
                (
                    name,
                    n,
                    missing,
                    missing,
                    "",
                    "pecora_continuity_statistic",
                    "missing_uzal_dimension"
                )
            )

            println(name, " -> missing uzal dimension")
            continue
        end

        dim, lags, status = pecora_nonuniform_lags(x, target_dim)

        push!(
            results,
            (
                name,
                n,
                target_dim,
                dim,
                lags,
                "pecora_continuity_statistic",
                status
            )
        )

        println(
            name,
            " -> target_dim=", target_dim,
            ", pecora_dim=", dim,
            ", lags=", lags,
            ", status=", status
        )
    end

    CSV.write(OUTPUT_PATH, results)

    println(results)
    println("Saved to: ", OUTPUT_PATH)
end


main()