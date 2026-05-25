# main.jl
using SymbolicRegression

# ==========================================
# ★追加: 極座標の運動項を再発見する
# ==========================================
sq(x) = x^2

# ==========================================
# 1. データ生成関数
# ==========================================
function generate_data(N::Int=200)
    r_data = rand(N) .* 4.5 .+ 0.5
    theta_data = rand(N) .* (2.0 * pi)
    dr_data = rand(N) .* 4.0 .- 2.0
    dtheta_data = rand(N) .* 4.0 .- 2.0

    X_train = Matrix(hcat(r_data, theta_data, dr_data, dtheta_data)')

    # 極座標の運動項: T = 1/2 (dr^2 + r^2 dtheta^2)
    T_data = 0.5 .* (sq.(dr_data) .+ sq.(r_data .* dtheta_data))

    return X_train, T_data
end

# ==========================================
# 2. 目的関数 (極座標の運動項のスコア)
# ==========================================
function action_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # 候補関数 T(r, θ, ṙ, θ̇) の評価
    f_res = eval_tree_array(tree, dataset.X, options)
    f_x = f_res[1]
    
    if f_x === nothing || any(x -> !isfinite(x), f_x)
        return L(Inf)
    end

    target_T = dataset.y

    mse_loss = sum((f_x .- target_T).^2) / length(target_T)
    
    if isnan(mse_loss) || isinf(mse_loss)
        return L(Inf)
    end
    
    return mse_loss
end

# ==========================================
# 3. 探索実行関数
# ==========================================
function run_search(X_train, L_data; niterations=100)
    options = Options(
        binary_operators=[+, -, *],
        unary_operators=[sq],
        loss_function=action_loss,
        npopulations=30,
        parsimony=0.1,
        verbosity=1,
        progress=true,
        early_stop_condition = (loss, complexity) -> loss < 1e-18 && complexity <= 9
    )
    
    hof = EquationSearch(X_train, L_data, options=options, niterations=niterations)
    return hof
end

function latest_hall_of_fame_path()
    if !isdir("outputs")
        return nothing
    end

    subdirs = filter(isdir, readdir("outputs"; join=true))
    isempty(subdirs) && return nothing

    latest_dir = argmax(path -> stat(path).mtime, subdirs)
    csv_path = joinpath(latest_dir, "hall_of_fame.csv")
    return isfile(csv_path) ? csv_path : nothing
end

function best_equation_from_hall_of_fame(csv_path::AbstractString)
    best_loss = Inf
    best_equation = ""

    for line in Iterators.drop(eachline(csv_path), 1)
        columns = split(line, ',', limit=3)
        length(columns) < 3 && continue

        loss = try
            parse(Float64, columns[2])
        catch
            continue
        end

        equation = strip(replace(columns[3], '"' => ' '))

        if loss < best_loss
            best_loss = loss
            best_equation = equation
        end
    end

    return best_loss, best_equation
end

pretty_number(x::Real) = begin
    rounded = round(Float64(x); digits=12)
    if isapprox(rounded, round(rounded); atol=1e-12, rtol=0)
        return string(Int(round(rounded)))
    end
    return string(rounded)
end

function normalize_equation_display(equation::AbstractString)
    cleaned = replace(equation, r"\s+" => " ")
    cleaned = replace(cleaned, "x4 * x1" => "x1 * x4")

    return cleaned
end

# ==========================================
# 4. 実行ブロック
# ==========================================
if !@isdefined(IS_TESTING)
    println("データを生成しています...")
    X_train, L_data = generate_data(200)
    
    println("極座標の運動項の探索を開始します...")
    global hof_result = run_search(X_train, L_data, niterations=40)
    
    println("\n探索が完了しました！")
    csv_path = latest_hall_of_fame_path()
    if csv_path !== nothing
        best_loss, best_equation = best_equation_from_hall_of_fame(csv_path)
        cleaned_equation = normalize_equation_display(best_equation)
        println("整理後の最良式: ", cleaned_equation)
        println("最良損失: ", best_loss)
    end
    # display(hof_result) 
end