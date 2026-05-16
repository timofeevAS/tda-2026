using CSV
using DataFrames
using DelayEmbeddings
using Statistics

const DATA_PATH = "dataset_clean.csv"
const MAX_TAU = 100
const MAX_DIMENSION = 12
const OUTPUT_PATH = "uzal_embedding_results_all_columns.csv"

df = CSV.read(DATA_PATH, DataFrame)
numeric_columns = names(df, Number)

results = DataFrame(
    column = String[],
    tau = Int[],
    dimension = Int[],
    uzal_cost = Float64[]
)

skipped = DataFrame(
    column = String[],
    reason = String[]
)

println("Numeric columns: ", length(numeric_columns))

for (column_index, column) in enumerate(numeric_columns)
    println("[", column_index, "/", length(numeric_columns), "] ", column)

    x = Float64.(collect(skipmissing(df[!, column])))

    if length(x) < 3 || std(x) == 0
        push!(skipped, (column, "not_enough_data_or_constant"))
        continue
    end

    valid_for_column = 0

    for tau in 1:MAX_TAU
        for dimension in 3:MAX_DIMENSION
            try
                embedded = embed(x, dimension, tau)
                cost = uzal_cost(embedded)

                push!(results, (column, tau, dimension, cost))
                valid_for_column += 1
            catch err
                # Some tau/dimension combinations are impossible for short series.
            end
        end
    end

    if valid_for_column == 0
        push!(skipped, (column, "no_valid_tau_dimension_combinations"))
    end
end

sort!(results, [:column, :uzal_cost])
CSV.write(OUTPUT_PATH, results)

println("Calculated combinations: ", nrow(results))
println("Skipped columns: ", nrow(skipped))
println("Saved results: ", OUTPUT_PATH)

if nrow(results) > 0
    println("Top 20 rows by uzal_cost across all columns:")
    best_results = first(sort(results, :uzal_cost), min(20, nrow(results)))
    show(best_results, allcols=true, allrows=true)
    println()
end

if nrow(skipped) > 0
    CSV.write("uzal_embedding_skipped_columns.csv", skipped)
    println("Saved skipped columns: uzal_embedding_skipped_columns.csv")
end
