using CSV
using DataFrames
using Statistics
using Random
using LinearAlgebra
using Printf


# =========================
# 1. НАСТРОЙКИ
# =========================

DATA_PATH = "dataset_clean_31.csv"

PECUZAL_PARAMS_PATH = "pecuzal_res.csv"
PECORA_PARAMS_PATH = "pecora_res.csv"

OUTPUT_FULL_PATH = "nonuniform_embedding_quality_full.csv"
OUTPUT_SUMMARY_PATH = "nonuniform_embedding_quality_summary.csv"
OUTPUT_BEST_PATH = "nonuniform_embedding_quality_best_by_column.csv"

HORIZON = 1
K_NEIGHBORS = 5
TEST_RATIO = 0.3

NOISE_LEVEL = 0.05
N_NOISE_RUNS = 10
DISTANCE_SAMPLE_SIZE = 500

RANDOM_SEED = 42


# =========================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

function clean_series(x)
    values = Float64[]

    for v in x
        if v === missing
            continue
        end

        fv = try
            Float64(v)
        catch
            NaN
        end

        if isfinite(fv)
            push!(values, fv)
        end
    end

    return values
end


function parse_lags(value)
    if value === missing
        return Int[]
    end

    s = strip(String(value))

    if isempty(s)
        return Int[]
    end

    parts = split(s, ";")

    lags = Int[]

    for p in parts
        p = strip(p)

        if isempty(p)
            continue
        end

        lag = try
            parse(Int, p)
        catch
            missing
        end

        if lag !== missing
            push!(lags, lag)
        end
    end

    # На всякий случай убираем повторы, но порядок сохраняем
    unique_lags = Int[]
    for lag in lags
        if !(lag in unique_lags)
            push!(unique_lags, lag)
        end
    end

    return unique_lags
end


function make_nonuniform_delay_embedding(series::Vector{Float64}, lags::Vector{Int}; horizon::Int=1)
    if isempty(lags)
        return nothing, nothing
    end

    n_total = length(series)

    max_lag = maximum(lags)
    target_shift = max_lag + horizon

    n_vectors = n_total - target_shift

    if n_vectors <= 0
        return nothing, nothing
    end

    dim = length(lags)
    X = Matrix{Float64}(undef, n_vectors, dim)

    for j in 1:dim
        lag = lags[j]
        X[:, j] = series[(1 + lag):(n_vectors + lag)]
    end

    y = series[(1 + target_shift):(n_vectors + target_shift)]

    return X, y
end


function standardize_train_test(X_train, X_test)
    means = vec(mean(X_train, dims=1))
    stds = vec(std(X_train, dims=1))

    stds = [s == 0.0 || !isfinite(s) ? 1.0 : s for s in stds]

    X_train_scaled = (X_train .- means') ./ stds'
    X_test_scaled = (X_test .- means') ./ stds'

    return X_train_scaled, X_test_scaled
end


function knn_predict_one(x_test, X_train, y_train, k::Int)
    n_train = size(X_train, 1)
    k_eff = min(k, n_train)

    distances = Vector{Float64}(undef, n_train)

    for i in 1:n_train
        distances[i] = norm(x_test .- X_train[i, :])
    end

    nearest_idx = partialsortperm(distances, 1:k_eff)

    nearest_distances = distances[nearest_idx]
    nearest_values = y_train[nearest_idx]

    eps = 1e-12

    if minimum(nearest_distances) < eps
        return nearest_values[argmin(nearest_distances)]
    end

    weights = 1.0 ./ (nearest_distances .+ eps)

    return sum(weights .* nearest_values) / sum(weights)
end


function knn_predict(X_test, X_train, y_train, k::Int)
    y_pred = Vector{Float64}(undef, size(X_test, 1))

    for i in 1:size(X_test, 1)
        y_pred[i] = knn_predict_one(X_test[i, :], X_train, y_train, k)
    end

    return y_pred
end


function reconstruction_error(series::Vector{Float64}, lags::Vector{Int};
                              horizon::Int=1,
                              k_neighbors::Int=5,
                              test_ratio::Float64=0.3)

    X, y = make_nonuniform_delay_embedding(series, lags; horizon=horizon)

    if X === nothing
        return (
            reconstruction_mae = NaN,
            reconstruction_rmse = NaN,
            reconstruction_nrmse = NaN
        )
    end

    n = size(X, 1)
    split_idx = floor(Int, n * (1.0 - test_ratio))

    if split_idx <= k_neighbors || split_idx >= n
        return (
            reconstruction_mae = NaN,
            reconstruction_rmse = NaN,
            reconstruction_nrmse = NaN
        )
    end

    X_train = X[1:split_idx, :]
    y_train = y[1:split_idx]

    X_test = X[(split_idx + 1):end, :]
    y_test = y[(split_idx + 1):end]

    X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

    y_pred = knn_predict(X_test_scaled, X_train_scaled, y_train, k_neighbors)

    errors = y_test .- y_pred

    mae = mean(abs.(errors))
    rmse = sqrt(mean(errors .^ 2))

    y_std = std(y_test)
    nrmse = y_std == 0.0 || !isfinite(y_std) ? NaN : rmse / y_std

    return (
        reconstruction_mae = mae,
        reconstruction_rmse = rmse,
        reconstruction_nrmse = nrmse
    )
end


function pairwise_distance_matrix(X::Matrix{Float64})
    n = size(X, 1)
    D = Matrix{Float64}(undef, n, n)

    for i in 1:n
        D[i, i] = 0.0

        for j in (i + 1):n
            d = norm(X[i, :] .- X[j, :])
            D[i, j] = d
            D[j, i] = d
        end
    end

    return D
end


function noise_error(series::Vector{Float64}, lags::Vector{Int};
                     noise_level::Float64=0.05,
                     n_noise_runs::Int=10,
                     distance_sample_size::Int=500,
                     random_seed::Int=42)

    rng = MersenneTwister(random_seed)

    series_std = std(series)

    if series_std == 0.0 || !isfinite(series_std)
        return (
            noise_point_shift = NaN,
            noise_distance_distortion = NaN
        )
    end

    X_clean, _ = make_nonuniform_delay_embedding(series, lags; horizon=1)

    if X_clean === nothing
        return (
            noise_point_shift = NaN,
            noise_distance_distortion = NaN
        )
    end

    n_points = size(X_clean, 1)
    dim = size(X_clean, 2)

    if n_points < 3
        return (
            noise_point_shift = NaN,
            noise_distance_distortion = NaN
        )
    end

    sample_size = min(distance_sample_size, n_points)
    sample_idx = randperm(rng, n_points)[1:sample_size]

    X_clean_sample = X_clean[sample_idx, :]
    D_clean = pairwise_distance_matrix(X_clean_sample)
    D_clean_norm = norm(D_clean)

    point_shifts = Float64[]
    distance_distortions = Float64[]

    for run in 1:n_noise_runs
        noise = noise_level * series_std .* randn(rng, length(series))
        noisy_series = series .+ noise

        X_noisy, _ = make_nonuniform_delay_embedding(noisy_series, lags; horizon=1)

        if X_noisy === nothing
            continue
        end

        if size(X_noisy) != size(X_clean)
            continue
        end

        shifts = [norm(X_clean[i, :] .- X_noisy[i, :]) for i in 1:n_points]

        normalized_shift = mean(shifts) / (series_std * sqrt(dim))

        X_noisy_sample = X_noisy[sample_idx, :]
        D_noisy = pairwise_distance_matrix(X_noisy_sample)

        if D_clean_norm == 0.0 || !isfinite(D_clean_norm)
            distortion = NaN
        else
            distortion = norm(D_clean .- D_noisy) / D_clean_norm
        end

        push!(point_shifts, normalized_shift)
        push!(distance_distortions, distortion)
    end

    if isempty(point_shifts)
        return (
            noise_point_shift = NaN,
            noise_distance_distortion = NaN
        )
    end

    return (
        noise_point_shift = mean(point_shifts),
        noise_distance_distortion = mean(distance_distortions)
    )
end


function minmax_normalize_column(v)
    valid = collect(skipmissing(v))
    valid = [x for x in valid if isfinite(x)]

    if isempty(valid)
        return fill(NaN, length(v))
    end

    vmin = minimum(valid)
    vmax = maximum(valid)

    if vmax == vmin
        return fill(0.0, length(v))
    end

    result = Float64[]

    for x in v
        if ismissing(x) || !isfinite(x)
            push!(result, NaN)
        else
            push!(result, (x - vmin) / (vmax - vmin))
        end
    end

    return result
end


# =========================
# 3. ЗАГРУЗКА ДАННЫХ
# =========================

data = CSV.read(DATA_PATH, DataFrame)

pecuzal_raw = CSV.read(PECUZAL_PARAMS_PATH, DataFrame)
pecora_raw = CSV.read(PECORA_PARAMS_PATH, DataFrame)


# =========================
# 4. ПРИВОДИМ PECUZAL К ОБЩЕМУ ФОРМАТУ
# =========================

pecuzal_params = DataFrame(
    column = String[],
    method = String[],
    dimension = Int[],
    lags = Vector{Int}[],
    lags_str = String[],
    cost = Union{Missing, Float64}[],
    status = String[]
)

for row in eachrow(pecuzal_raw)
    status = String(row.status)

    if status != "ok"
        continue
    end

    col_name = String(row.timeseries_name)
    lags = parse_lags(row.pecuzal_lags)

    if isempty(lags)
        continue
    end

    dim = length(lags)

    cost_value = if row.pecuzal_cost === missing
        missing
    else
        Float64(row.pecuzal_cost)
    end

    push!(
        pecuzal_params,
        (
            col_name,
            "pecuzal",
            dim,
            lags,
            join(lags, ";"),
            cost_value,
            status
        )
    )
end


# =========================
# 5. ПРИВОДИМ PECORA К ОБЩЕМУ ФОРМАТУ
# =========================

pecora_params = DataFrame(
    column = String[],
    method = String[],
    dimension = Int[],
    lags = Vector{Int}[],
    lags_str = String[],
    cost = Union{Missing, Float64}[],
    status = String[]
)

for row in eachrow(pecora_raw)
    status = String(row.status)

    if status != "ok"
        continue
    end

    col_name = String(row.timeseries_name)
    lags = parse_lags(row.pecora_lags)

    if isempty(lags)
        continue
    end

    dim = length(lags)

    push!(
        pecora_params,
        (
            col_name,
            "pecora",
            dim,
            lags,
            join(lags, ";"),
            missing,
            status
        )
    )
end


# =========================
# 6. СРАВНИВАЕМ ТОЛЬКО ОБЩИЕ РЯДЫ
# =========================

common_columns = intersect(
    Set(pecuzal_params.column),
    Set(pecora_params.column)
)

pecuzal_params = filter(row -> row.column in common_columns, pecuzal_params)
pecora_params = filter(row -> row.column in common_columns, pecora_params)

params = vcat(pecuzal_params, pecora_params)

println("Рядов в Pecuzal: ", length(unique(pecuzal_params.column)))
println("Рядов в Pecora: ", length(unique(pecora_params.column)))
println("Общих рядов: ", length(common_columns))
println("Всего строк параметров: ", nrow(params))


# =========================
# 7. ОЦЕНКА КАЧЕСТВА
# =========================

results = DataFrame(
    column = String[],
    method = String[],
    dimension = Int[],
    lags_str = String[],
    cost = Union{Missing, Float64}[],
    n_observations = Int[],
    reconstruction_mae = Float64[],
    reconstruction_rmse = Float64[],
    reconstruction_nrmse = Float64[],
    noise_point_shift = Float64[],
    noise_distance_distortion = Float64[]
)

for row in eachrow(params)
    col_name = row.column
    method = row.method
    dim = row.dimension
    lags = row.lags

    if !(col_name in names(data))
        @warn "Колонки нет в data, пропускаю" col_name
        continue
    end

    series = clean_series(data[!, col_name])

    max_lag = maximum(lags)

    if length(series) < max_lag + HORIZON + 20
        @warn "Слишком короткий ряд для таких lags, пропускаю" col_name lags length(series)
        continue
    end

    rec = reconstruction_error(
        series,
        lags;
        horizon = HORIZON,
        k_neighbors = K_NEIGHBORS,
        test_ratio = TEST_RATIO
    )

    noise = noise_error(
        series,
        lags;
        noise_level = NOISE_LEVEL,
        n_noise_runs = N_NOISE_RUNS,
        distance_sample_size = DISTANCE_SAMPLE_SIZE,
        random_seed = RANDOM_SEED
    )

    push!(
        results,
        (
            col_name,
            method,
            dim,
            row.lags_str,
            row.cost,
            length(series),
            rec.reconstruction_mae,
            rec.reconstruction_rmse,
            rec.reconstruction_nrmse,
            noise.noise_point_shift,
            noise.noise_distance_distortion
        )
    )

    @printf(
        "Готово: %-25s %-8s dim=%2d lags=%-20s | NRMSE=%.4f | noise_dist=%.4f\n",
        col_name,
        method,
        dim,
        row.lags_str,
        rec.reconstruction_nrmse,
        noise.noise_distance_distortion
    )
end


# =========================
# 8. QUALITY SCORE
# =========================

results.reconstruction_nrmse_norm = minmax_normalize_column(results.reconstruction_nrmse)
results.noise_point_shift_norm = minmax_normalize_column(results.noise_point_shift)
results.noise_distance_distortion_norm = minmax_normalize_column(results.noise_distance_distortion)

quality_score = Float64[]

for row in eachrow(results)
    vals = [
        row.reconstruction_nrmse_norm,
        row.noise_point_shift_norm,
        row.noise_distance_distortion_norm
    ]

    vals = [v for v in vals if isfinite(v)]

    if isempty(vals)
        push!(quality_score, NaN)
    else
        push!(quality_score, mean(vals))
    end
end

results.quality_score = quality_score

sort!(results, [:column, :quality_score])


# =========================
# 9. SUMMARY
# =========================

summary = combine(
    groupby(results, :method),
    :reconstruction_nrmse => mean => :mean_reconstruction_nrmse,
    :noise_point_shift => mean => :mean_noise_point_shift,
    :noise_distance_distortion => mean => :mean_noise_distance_distortion,
    :quality_score => mean => :mean_quality_score,
    nrow => :n_rows
)

sort!(summary, :mean_quality_score)


# =========================
# 10. СОХРАНЕНИЕ
# =========================

CSV.write(OUTPUT_FULL_PATH, results)
CSV.write(OUTPUT_SUMMARY_PATH, summary)

println()
println("=== SUMMARY BY METHOD ===")
show(summary, allrows=true, allcols=true)

println()
println()
println("Файл с полными результатами сохранен в: ", OUTPUT_FULL_PATH)
println("Файл со сводкой сохранен в: ", OUTPUT_SUMMARY_PATH)


# =========================
# 11. ЛУЧШИЙ МЕТОД ДЛЯ КАЖДОГО РЯДА
# =========================

best_by_column = combine(
    groupby(results, :column)
) do sdf
    sdf_sorted = sort(sdf, :quality_score)
    first(sdf_sorted, 1)
end

CSV.write(OUTPUT_BEST_PATH, best_by_column)

println()
println("=== BEST METHOD BY COLUMN ===")
show(best_by_column[:, [:column, :method, :dimension, :lags_str, :quality_score]], allrows=true, allcols=true)

println()
println()
println("Лучшие методы по каждому ряду сохранены в: ", OUTPUT_BEST_PATH)