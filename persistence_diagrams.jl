using CSV
using DataFrames
using DelayEmbeddings
using Statistics
using Ripserer

const DATA_PATH = "dataset_clean.csv"

# Файл с лучшими tau и dimension после критерия Узала.
const BEST_PARAMS_PATH = "uniform_dimensions.csv"

const OUTPUT_PATH = "persistence_diagrams_all_columns.csv"
const SKIPPED_PATH = "persistence_diagrams_skipped_columns.csv"

# Считаем H0 и H1.
# H0 — компоненты связности.
# H1 — циклы / петли.
const DIM_MAX = 1

# Ограничение числа точек в embedding.
# Диаграммы персистентности могут считаться долго, если точек слишком много.
const MAX_POINTS = 1200


function normalize_series(x)
    x = collect(skipmissing(x))
    x = Float64.(x)

    # Убираем NaN и Inf, если они вдруг есть
    x = x[isfinite.(x)]

    if length(x) < 10
        return nothing
    end

    s = std(x)

    if !isfinite(s) || s == 0
        return nothing
    end

    return (x .- mean(x)) ./ s
end


function subsample_points(points, max_points)
    n = length(points)

    if n <= max_points
        return points
    end

    # Берем равномерную подвыборку по времени
    idx = unique(round.(Int, range(1, n, length=max_points)))

    return points[idx]
end


function embedding_to_points(embedded)
    # embed(...) возвращает набор точек фазового пространства.
    # Каждую точку переводим в обычный Vector{Float64},
    # чтобы Ripserer мог считать по облаку точек.
    return [collect(embedded[i]) for i in 1:length(embedded)]
end


println("Reading dataset: ", DATA_PATH)
df = CSV.read(DATA_PATH, DataFrame)

println("Reading best embedding parameters: ", BEST_PARAMS_PATH)
best_params = CSV.read(BEST_PARAMS_PATH, DataFrame)

required_columns = ["column", "tau", "dimension"]
missing_columns = setdiff(required_columns, names(best_params))

if !isempty(missing_columns)
    error("В файле с лучшими параметрами нет колонок: $(missing_columns)")
end

results = DataFrame(
    column = String[],
    tau = Int[],
    dimension = Int[],
    homology_dimension = Int[],
    birth = Float64[],
    death = Float64[],
    persistence = Float64[],
    is_infinite = Bool[]
)

skipped = DataFrame(
    column = String[],
    reason = String[]
)

println("Rows in best parameters table: ", nrow(best_params))

for (i, row) in enumerate(eachrow(best_params))
    column_name = String(row.column)

    println("[", i, "/", nrow(best_params), "] ", column_name)

    if !(column_name in names(df))
        push!(skipped, (column_name, "column_not_found_in_dataset"))
        continue
    end

    tau = Int(round(row.tau))
    dimension = Int(round(row.dimension))

    x = normalize_series(df[!, column_name])

    if x === nothing
        push!(skipped, (column_name, "not_enough_data_or_constant_or_invalid"))
        continue
    end

    try
        embedded = embed(x, dimension, tau)
        points = embedding_to_points(embedded)
        points = subsample_points(points, MAX_POINTS)

        if length(points) < dimension + 2
            push!(skipped, (column_name, "too_few_embedding_points"))
            continue
        end

        diagrams = ripserer(points; dim_max=DIM_MAX)

        for dim_index in eachindex(diagrams)
            homology_dim = dim_index - 1
            diagram = diagrams[dim_index]

            for interval in diagram
                b = Float64(birth(interval))
                d = Float64(death(interval))

                is_inf = !isfinite(d)
                pers = is_inf ? Inf : d - b

                push!(
                    results,
                    (
                        column_name,
                        tau,
                        dimension,
                        homology_dim,
                        b,
                        d,
                        pers,
                        is_inf
                    )
                )
            end
        end

    catch err
        push!(skipped, (column_name, string(err)))
        continue
    end
end

CSV.write(OUTPUT_PATH, results)
CSV.write(SKIPPED_PATH, skipped)

println("Saved persistence diagrams to: ", OUTPUT_PATH)
println("Saved skipped columns to: ", SKIPPED_PATH)
println("Total diagram points: ", nrow(results))
println("Skipped columns: ", nrow(skipped))

if nrow(results) > 0
    println("First rows:")
    show(first(results, min(10, nrow(results))), allcols=true, allrows=true)
    println()
end