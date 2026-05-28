using CSV
using DataFrames
using Statistics
using LinearAlgebra

const DATA_PATH = "dataset_clean.csv"
const OUTPUT_PATH = "results.csv"

const HARD_MAX_TAU = 200
const AMI_BINS = 16
const TIME_COL = "time"


# ============================================================
# Helpers
# ============================================================

function first_local_min(v::Vector{Float64})
    if length(v) < 3
        return missing
    end

    for i in 2:length(v)-1
        if isfinite(v[i]) && v[i] < v[i - 1] && v[i] < v[i + 1]
            return i
        end
    end

    return missing
end


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


function choose_max_tau(n::Int)
    return max(1, min(HARD_MAX_TAU, n ÷ 10))
end


# ============================================================
# ACF
# ============================================================

function acf_values(x::Vector{Float64}, max_tau::Int)
    n = length(x)

    if n < 3
        return Float64[]
    end

    max_tau = min(max_tau, n - 2)

    μ = mean(x)
    y = x .- μ

    denom = dot(y, y)

    if denom == 0
        return fill(NaN, max_tau)
    end

    acf = Float64[]

    for τ in 1:max_tau
        v = dot(y[1:n-τ], y[1+τ:n]) / denom
        push!(acf, v)
    end

    return acf
end


function tau_acf(x::Vector{Float64}, max_tau::Int)
    if length(x) < 20
        return missing
    end

    acf = acf_values(x, max_tau)

    if isempty(acf)
        return missing
    end

    # 1. Первый локальный минимум
    τ_min = first_local_min(acf)

    if τ_min !== missing
        return τ_min
    end

    # 2. Первое падение ниже 1/e
    threshold = 1 / exp(1)

    for i in eachindex(acf)
        if isfinite(acf[i]) && acf[i] <= threshold
            return i
        end
    end

    # 3. Fallback: глобальный минимум
    finite_idx = findall(isfinite, acf)

    if isempty(finite_idx)
        return missing
    end

    return finite_idx[argmin(acf[finite_idx])]
end


# ============================================================
# AMI
# ============================================================

function discretize_equal_width(x::Vector{Float64}, bins::Int)
    xmin = minimum(x)
    xmax = maximum(x)

    if xmin == xmax
        return ones(Int, length(x))
    end

    edges = range(xmin, xmax; length=bins + 1)

    result = Vector{Int}(undef, length(x))

    for i in eachindex(x)
        b = searchsortedlast(edges, x[i])

        if b < 1
            b = 1
        elseif b > bins
            b = bins
        end

        result[i] = b
    end

    return result
end


function mutual_information_discrete(a::Vector{Int}, b::Vector{Int}, bins::Int)
    n = length(a)

    joint = zeros(Float64, bins, bins)
    pa = zeros(Float64, bins)
    pb = zeros(Float64, bins)

    for i in 1:n
        ai = a[i]
        bi = b[i]

        joint[ai, bi] += 1
        pa[ai] += 1
        pb[bi] += 1
    end

    joint ./= n
    pa ./= n
    pb ./= n

    mi = 0.0

    for i in 1:bins
        for j in 1:bins
            if joint[i, j] > 0 && pa[i] > 0 && pb[j] > 0
                mi += joint[i, j] * log(joint[i, j] / (pa[i] * pb[j]))
            end
        end
    end

    return mi
end


function ami_values(x::Vector{Float64}, max_tau::Int, bins::Int)
    n = length(x)

    if n < 3
        return Float64[]
    end

    max_tau = min(max_tau, n - 2)

    xd = discretize_equal_width(x, bins)

    ami = Float64[]

    for τ in 1:max_tau
        a = xd[1:n-τ]
        b = xd[1+τ:n]

        push!(ami, mutual_information_discrete(a, b, bins))
    end

    return ami
end


function tau_ami(x::Vector{Float64}, max_tau::Int, bins::Int)
    if length(x) < 20
        return missing
    end

    ami = ami_values(x, max_tau, bins)

    if isempty(ami)
        return missing
    end

    # 1. Первый локальный минимум AMI
    τ_min = first_local_min(ami)

    if τ_min !== missing
        return τ_min
    end

    # 2. Fallback: глобальный минимум AMI
    finite_idx = findall(isfinite, ami)

    if isempty(finite_idx)
        return missing
    end

    return finite_idx[argmin(ami[finite_idx])]
end


# ============================================================
# Main
# ============================================================

function main()
    df = CSV.read(DATA_PATH, DataFrame)

    results = DataFrame(
        timeseries_name = String[],
        n = Int[],
        max_tau = Int[],
        acf_tau = Union{Missing, Int}[],
        ami_tau = Union{Missing, Int}[],
        selected_tau = Union{Missing, Int}[]
    )

    for col in names(df)
        if col == TIME_COL
            continue
        end

        x = safe_float_vector(df[:, col])
        n = length(x)

        if n < 20
            push!(results, (string(col), n, 0, missing, missing, missing))
            continue
        end

        max_tau = choose_max_tau(n)

        acfτ = tau_acf(x, max_tau)
        amiτ = tau_ami(x, max_tau, AMI_BINS)

        selectedτ = amiτ !== missing ? amiτ : acfτ

        push!(results, (string(col), n, max_tau, acfτ, amiτ, selectedτ))
    end

    CSV.write(OUTPUT_PATH, results)

    println(results)
    println("Saved to: ", OUTPUT_PATH)
end


main()