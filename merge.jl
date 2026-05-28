using CSV
using DataFrames

const FNN_ACF_PATH = "fnn_embedding_results_all_columns.csv"
const UZAL_PATH = "uzal_cost_final_result.csv"
const OUTPUT_PATH = "merged_tau_dim_results.csv"

function main()
    fnn_acf = CSV.read(FNN_ACF_PATH, DataFrame)
    uzal = CSV.read(UZAL_PATH, DataFrame)

    # Переименовываем колонки, чтобы они совпадали по смыслу
    rename!(fnn_acf, :timeseries_name => :name)
    rename!(fnn_acf, :tau => :tau_acf)

    rename!(uzal, :column => :name)
    rename!(uzal, :tau => :uzal_tau)
    rename!(uzal, :dimension => :uzal_dim)

    # Оставляем только нужные колонки
    fnn_acf = fnn_acf[:, [:name, :tau_acf, :fnn_dim]]
    uzal = uzal[:, [:name, :uzal_tau, :uzal_dim]]

    # leftjoin: сохраняем все ряды из FNN/ACF файла
    result = leftjoin(fnn_acf, uzal, on=:name)

    # Порядок колонок как ты просил
    result = result[:, [:name, :tau_acf, :fnn_dim, :uzal_tau, :uzal_dim]]

    CSV.write(OUTPUT_PATH, result)

    println(result)
    println("Saved to: ", OUTPUT_PATH)
end

main()