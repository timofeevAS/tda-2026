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

ACF_FNN_PARAMS_PATH = "embedding_params_31_acf_fnn.csv"
UZAL_PARAMS_PATH = "uzal_cost_final_result.csv"

OUTPUT_FULL_PATH = "embedding_quality_comparison_full.csv"
OUTPUT_SUMMARY_PATH = "embedding_quality_comparison_summary.csv"

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


function make_delay_embedding(series::Vector{Float64}, tau::Int, dim::Int; horizon::Int=1)
    n_total = length(series)
    max_shift = (dim - 1) * tau
    target_shift = max_shift + horizon
    n_vectors = n_total - target_shift

    if n_vectors <= 0
        return nothing, nothing
    end

    X = Matrix{Float64}(undef, n_vectors, dim)

    for j in 1:dim
        shift = (j - 1) * tau
        X[:, j] = series[(1 + shift):(n_vectors + shift)]
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


function reconstruction_error(series::Vector{Float64}, tau::Int, dim::Int;
                              horizon::Int=1,
                              k_neighbors::Int=5,
                              test_ratio::Float64=0.3)

    X, y = make_delay_embedding(series, tau, dim; horizon=horizon)

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


function noise_error(series::Vector{Float64}, tau::Int, dim::Int;
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

    X_clean, _ = make_delay_embedding(series, tau, dim; horizon=1)

    if X_clean === nothing
        return (
            noise_point_shift = NaN,
            noise_distance_distortion = NaN
        )
    end

    n_points = size(X_clean, 1)

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

        X_noisy, _ = make_delay_embedding(noisy_series, tau, dim; horizon=1)

        if X_noisy === nothing
            continue
        end

        if size(X_noisy) != size(X_clean)
            continue
        end

        shifts = [norm(X_clean[i, :] .- X_noisy[i, :]) for i in 1:n_points]

        # Нормируем на std ряда и sqrt(dim), чтобы не штрафовать большую dimension
        # просто за то, что координат больше.
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

acf_fnn_params = CSV.read(ACF_FNN_PARAMS_PATH, DataFrame)
uzal_params = CSV.read(UZAL_PARAMS_PATH, DataFrame)

# Приводим имена рядов к строкам
acf_fnn_params.column = String.(acf_fnn_params.column)
uzal_params.column = String.(uzal_params.column)

# Берем список рядов из ACF+FNN
acf_columns = Set(acf_fnn_params.column)

# Оставляем в Uzal только те ряды, которые есть в ACF+FNN
uzal_params = filter(row -> row.column in acf_columns, uzal_params)

# Если в Uzal несколько вариантов для одного ряда,
# оставляем один лучший вариант.
# Так как Uzal cost минимизируется, берем строку с минимальным uzal_cost.
uzal_params = combine(groupby(uzal_params, :column)) do sdf
    sdf_sorted = sort(sdf, :uzal_cost)
    first(sdf_sorted, 1)
end

# Если в ACF+FNN вдруг тоже есть дубли, оставляем первый вариант
acf_fnn_params = combine(groupby(acf_fnn_params, :column)) do sdf
    first(sdf, 1)
end

acf_fnn_params.method .= "acf_fnn"
uzal_params.method .= "uzal"

params = vcat(
    acf_fnn_params[:, [:column, :tau, :dimension, :uzal_cost, :method]],
    uzal_params[:, [:column, :tau, :dimension, :uzal_cost, :method]]
)

params.tau = Int.(params.tau)
params.dimension = Int.(params.dimension)

println("Рядов в ACF+FNN: ", length(unique(acf_fnn_params.column)))
println("Рядов в Uzal после фильтрации и удаления дублей: ", length(unique(uzal_params.column)))
println("Всего строк параметров: ", nrow(params))
println("Количество колонок в data: ", ncol(data))

missing_in_uzal = setdiff(
    Set(acf_fnn_params.column),
    Set(uzal_params.column)
)

println("Ряды из ACF+FNN, которых нет в Uzal: ", collect(missing_in_uzal))


# =========================
# 4. ОЦЕНКА КАЧЕСТВА ВЛОЖЕНИЙ
# =========================

results = DataFrame(
    column = String[],
    method = String[],
    tau = Int[],
    dimension = Int[],
    uzal_cost = Union{Missing, Float64}[],
    n_observations = Int[],
    reconstruction_mae = Float64[],
    reconstruction_rmse = Float64[],
    reconstruction_nrmse = Float64[],
    noise_point_shift = Float64[],
    noise_distance_distortion = Float64[]
)

for row in eachrow(params)
    col_name = String(row.column)
    method = String(row.method)
    tau = Int(row.tau)
    dim = Int(row.dimension)

    if !(col_name in names(data))
        @warn "Колонки нет в data, пропускаю" col_name
        continue
    end

    series = clean_series(data[!, col_name])

    if length(series) < (dim - 1) * tau + HORIZON + 20
        @warn "Слишком короткий ряд для такого tau/dimension, пропускаю" col_name tau dim length(series)
        continue
    end

    rec = reconstruction_error(
        series,
        tau,
        dim;
        horizon = HORIZON,
        k_neighbors = K_NEIGHBORS,
        test_ratio = TEST_RATIO
    )

    noise = noise_error(
        series,
        tau,
        dim;
        noise_level = NOISE_LEVEL,
        n_noise_runs = N_NOISE_RUNS,
        distance_sample_size = DISTANCE_SAMPLE_SIZE,
        random_seed = RANDOM_SEED
    )

    uzal_cost_value = if ismissing(row.uzal_cost)
        missing
    else
        Float64(row.uzal_cost)
    end

    push!(
        results,
        (
            col_name,
            method,
            tau,
            dim,
            uzal_cost_value,
            length(series),
            rec.reconstruction_mae,
            rec.reconstruction_rmse,
            rec.reconstruction_nrmse,
            noise.noise_point_shift,
            noise.noise_distance_distortion
        )
    )

    @printf(
        "Готово: %-25s %-8s tau=%3d dim=%2d | NRMSE=%.4f | noise_dist=%.4f\n",
        col_name,
        method,
        tau,
        dim,
        rec.reconstruction_nrmse,
        noise.noise_distance_distortion
    )
end


# =========================
# 5. ОБЩИЙ SCORE
# =========================
#
# Все три метрики ниже должны быть маленькими:
# - reconstruction_nrmse
# - noise_point_shift
# - noise_distance_distortion
#
# Поэтому нормируем их в диапазон 0..1 и усредняем.
# Чем меньше quality_score, тем лучше вложение.

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
# 6. СРАВНЕНИЕ МЕТОДОВ
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
# 7. СОХРАНЕНИЕ
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
# 8. ЛУЧШИЙ МЕТОД ДЛЯ КАЖДОГО РЯДА
# =========================

best_by_column = combine(
    groupby(results, :column)
) do sdf
    sdf_sorted = sort(sdf, :quality_score)
    first(sdf_sorted, 1)
end

println()
println("=== BEST METHOD BY COLUMN ===")
show(best_by_column[:, [:column, :method, :tau, :dimension, :quality_score]], allrows=true, allcols=true)

CSV.write("embedding_quality_best_by_column.csv", best_by_column)
println()
println()
println("Лучшие методы по каждому ряду сохранены в: embedding_quality_best_by_column.csv")