# main.jl
using SymbolicRegression

# ==========================================
# ★追加: 安全な自作関数を使って、極座標の半径を再発見する
# ==========================================
sq(x) = x^2
safe_sqrt(x) = sqrt(abs(x))

# ==========================================
# 1. データ生成関数
# ==========================================
function generate_data(N::Int=200)
    x_data = rand(N) .* 10.0 .- 5.0
    y_data = rand(N) .* 10.0 .- 5.0

    X_train = Matrix(hcat(x_data, y_data)')

    # 極座標の半径 r = sqrt(x^2 + y^2)
    r_data = safe_sqrt.(sq.(x_data) .+ sq.(y_data))

    return X_train, r_data
end

# ==========================================
# 2. 目的関数 (極座標の半径のスコア)
# ==========================================
function action_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # 候補関数 r(x, y) の評価
    f_res = eval_tree_array(tree, dataset.X, options)
    f_x = f_res[1]
    
    if f_x === nothing || any(x -> !isfinite(x), f_x)
        return L(Inf)
    end

    target_r = dataset.y

    mse_loss = sum((f_x .- target_r).^2) / length(target_r)
    
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
        unary_operators=[sq, safe_sqrt],
        loss_function=action_loss,
        npopulations=30,
        parsimony=0.1,
        verbosity=1,
        progress=true,
        early_stop_condition = (loss, complexity) -> loss < 1e-10 && complexity < 5
    )
    
    hof = EquationSearch(X_train, L_data, options=options, niterations=niterations)
    return hof
end

# ==========================================
# 4. 実行ブロック
# ==========================================
if !@isdefined(IS_TESTING)
    println("データを生成しています...")
    X_train, L_data = generate_data(200)
    
    println("極座標の半径の探索を開始します...")
    global hof_result = run_search(X_train, L_data, niterations=40)
    
    println("\n探索が完了しました！")
    # display(hof_result) 
end