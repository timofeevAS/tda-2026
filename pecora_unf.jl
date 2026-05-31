using CSV
using DataFrames
using DelayEmbeddings
using Random
using Statistics

const DATA_PATH = "dataset_clean.csv"
const EMBEDDING_PARAMS_PATH = "acf_fnn_emb.csv"
const OUTPUT_PATH = "pecora_unf_res.csv"

const MIN_N = 50
const MIN_DELAY = 10
const MAX_DELAY = 100
const THEILER_WINDOW = 1
const SAMPLESIZE = 0.5
const K_NEIGHBORS = 13
const RANDOM_SEED = 42
const JITTER_LEVEL = 1e-10


function safe_float_vector(values)
    result = Float64[]

    for value in values
        if ismissing(value)
            continue
        end

        try
            x = Float64(value)
            if isfinite(x)
                push!(result, x)
            end
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

    y = (x .- mean(x)) ./ σ

    rng = MersenneTwister(RANDOM_SEED)
    return y .+ JITTER_LEVEL .* randn(rng, length(y))
end


function required_column(df::DataFrame, candidates::Vector{String})
    available = Set(names(df))

    for candidate in candidates
        if candidate in available
            return candidate
        end
    end

    error("Missing required column. Expected one of: " * join(candidates, ", "))
end


function first_local_maximum_index(scores::Vector{Float64})
    if length(scores) < 3
        return nothing
    end

    for i in 2:(length(scores) - 1)
        if scores[i] >= scores[i - 1] && scores[i] >= scores[i + 1]
            return i
        end
    end

    return nothing
end


function choose_next_delay(scores::Vector{Float64}, delays::Vector{Int}, selected_lags::Vector{Int})
    selected = Set(selected_lags)
    valid = [
        i for i in eachindex(delays)
        if isfinite(scores[i]) && !(delays[i] in selected)
    ]

    if isempty(valid)
        return nothing
    end

    local_scores = fill(-Inf, length(scores))
    local_scores[valid] .= scores[valid]
    local_idx = first_local_maximum_index(local_scores)

    if local_idx !== nothing && local_idx in valid
        return delays[local_idx]
    end

    best_idx = valid[argmax(scores[valid])]
    return delays[best_idx]
end


function pecora_lags(x::Vector{Float64}, target_dim::Int)
    n = length(x)

    if n < MIN_N
        return missing, "", "too_short"
    end

    if target_dim < 1
        return missing, "", "bad_target_dim"
    end

    x = normalize_series(x)
    tmax = min(MAX_DELAY, n ÷ 10)

    if tmax < MIN_DELAY
        return missing, "", "bad_tmax"
    end

    delays = collect(MIN_DELAY:tmax)
    selected_lags = [0]

    try
        while length(selected_lags) < target_dim
            Random.seed!(RANDOM_SEED + length(selected_lags))

            ε★, _ = pecora(
                x,
                Tuple(selected_lags);
                delays = delays,
                w = THEILER_WINDOW,
                samplesize = SAMPLESIZE,
                K = K_NEIGHBORS,
            )

            scores = vec(ε★)
            next_lag = choose_next_delay(scores, delays, selected_lags)

            if next_lag === nothing
                return missing, join(selected_lags, ";"), "no_valid_next_delay"
            end

            push!(selected_lags, next_lag)
        end

        return length(selected_lags), join(selected_lags, ";"), "ok"
    catch err
        return missing, join(selected_lags, ";"), "error: " * sprint(showerror, err)
    end
end


function main()
    data = CSV.read(DATA_PATH, DataFrame)
    params = CSV.read(EMBEDDING_PARAMS_PATH, DataFrame)

    name_col = required_column(params, ["column", "timeseries_name"])
    dim_col = required_column(params, ["dimension", "target_dim"])

    results = DataFrame(
        timeseries_name = String[],
        n = Int[],
        target_dim = Union{Missing, Int}[],
        pecora_dim = Union{Missing, Int}[],
        pecora_lags = String[],
        method = String[],
        status = String[],
    )

    for row in eachrow(params)
        name = string(row[name_col])

        if !(name in names(data))
            push!(results, (name, 0, missing, missing, "", "pecora_continuity_statistic_acf_fnn_dim", "missing_timeseries"))
            println(name, " -> status=missing_timeseries")
            continue
        end

        target_dim = try
            Int(row[dim_col])
        catch
            missing
        end

        x = safe_float_vector(data[!, name])
        n = length(x)

        if ismissing(target_dim)
            push!(results, (name, n, missing, missing, "", "pecora_continuity_statistic_acf_fnn_dim", "missing_target_dim"))
            println(name, " -> status=missing_target_dim")
            continue
        end

        pecora_dim, lags, status = pecora_lags(x, target_dim)

        push!(
            results,
            (
                name,
                n,
                target_dim,
                pecora_dim,
                lags,
                "pecora_continuity_statistic_acf_fnn_dim",
                status,
            ),
        )

        println(
            name,
            " -> target_dim=", target_dim,
            ", pecora_dim=", pecora_dim,
            ", lags=", lags,
            ", status=", status,
        )
    end

    CSV.write(OUTPUT_PATH, results)
    println("Saved to: ", OUTPUT_PATH)
end


main()
