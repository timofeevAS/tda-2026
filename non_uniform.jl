using CSV
using DataFrames
using Statistics
using DelayEmbeddings

const DATA_PATH = "dataset_clean.csv"
const OUTPUT_PATH = "pecuzal_res.csv"

const TIME_COL = "time"

const MIN_N = 50
const TMAX = 100
const THEILER_WINDOW = 1
const ECON = true


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


function pecuzal_lags(x::Vector{Float64})
    n = length(x)

    if n < MIN_N
        return missing, "", missing, "too_short"
    end

    x = normalize_series(x)

    # Не берём слишком большой максимум лагов.
    # Для n = 1439 это даст 100.
    tmax = min(TMAX, n ÷ 10)

    if tmax < 2
        return missing, "", missing, "bad_tmax"
    end

    try
        Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(
            x;
            τs = 0:tmax,
            w = THEILER_WINDOW,
            econ = ECON
        )

        lags = Int.(collect(τ_vals))

        if !(0 in lags)
            lags = vcat(0, lags)
        end

        lags = unique(lags)
        sort!(lags)

        dim = length(lags)
        lags_str = join(lags, ";")
        cost = sum(Ls)

        return dim, lags_str, cost, "ok"

    catch err
        return missing, "", missing, "error: " * sprint(showerror, err)
    end
end


function main()
    df = CSV.read(DATA_PATH, DataFrame)

    results = DataFrame(
        timeseries_name = String[],
        n = Int[],
        pecuzal_dim = Union{Missing, Int}[],
        pecuzal_lags = String[],
        pecuzal_cost = Union{Missing, Float64}[],
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

        dim, lags, cost, status = pecuzal_lags(x)

        push!(
            results,
            (
                name,
                n,
                dim,
                lags,
                cost,
                "pecuzal",
                status
            )
        )

        println(
            name,
            " -> dim=", dim,
            ", lags=", lags,
            ", cost=", cost,
            ", status=", status
        )
    end

    CSV.write(OUTPUT_PATH, results)

    println(results)
    println("Saved to: ", OUTPUT_PATH)
end


main()