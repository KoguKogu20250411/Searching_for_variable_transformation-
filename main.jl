# main.jl
using SymbolicRegression

# ==========================================
# 1. データ生成関数
# ==========================================
function generate_data(N::Int=200)
    x_data = rand(N) .* 10.0 .- 5.0
    v_data = rand(N) .* 10.0 .- 5.0
    
    L_data = 0.5 .* v_data.^2 .- (0.5 .* x_data.^2 .- 3.0 .* x_data)
    X_train = Matrix(hcat(x_data, v_data)') 
    
    return X_train, L_data
end

# ==========================================
# 2. 目的関数 (二次形式化のスコア)
# ==========================================
function action_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # ① 候補関数 f(x) の評価
    f_res = eval_tree_array(tree, dataset.X, options)
    f_x = f_res[1]
    
    if f_x === nothing || any(x -> !isfinite(x), f_x)
        return L(Inf)
    end
    
    # ② 微分 f'(x) を「数値微分 (Finite Difference)」で計算する！
    # （Zygoteによる自動微分の衝突バグを完全に回避します）
    dx = 1e-4
    
    # x座標（1行目）だけを微小に動かしたデータを作る
    X_plus = copy(dataset.X)
    X_plus[1, :] .+= dx
    
    X_minus = copy(dataset.X)
    X_minus[1, :] .-= dx
    
    # 微小に動かした位置での f(x) を計算
    f_plus_res = eval_tree_array(tree, X_plus, options)
    f_minus_res = eval_tree_array(tree, X_minus, options)
    
    if f_plus_res[1] === nothing || f_minus_res[1] === nothing
        return L(Inf)
    end
    
    # 傾きを計算
    df_dx = (f_plus_res[1] .- f_minus_res[1]) ./ (2.0 * dx)
    
    if any(x -> !isfinite(x), df_dx)
        return L(Inf)
    end
    
    v = dataset.X[2, :]
    L_target = dataset.y
    
    # ③ 変換後の作用
    L_pred_base = 0.5 .* (df_dx .* v).^2 .- 0.5 .* f_x.^2
    
    # ④ 定数のズレ(C)を自動で吸収して誤差を計算
    C = sum(L_target .- L_pred_base) / length(L_target)
    L_pred = L_pred_base .+ C
    
    mse_loss = sum((L_pred .- L_target).^2) / length(L_target)
    
    if isnan(mse_loss) || isinf(mse_loss)
        return L(Inf)
    end
    
    return mse_loss
end

# ==========================================
# 3. 探索実行関数
# ==========================================
function run_search(X_train, L_data; niterations=40)
    options = Options(
        binary_operators=[+, -, *, /],
        loss_function=action_loss,
        npopulations=30,
        parsimony=0.01,
        verbosity=1,      # 1にすることでエラー回避しつつ静かにする
        progress=true     # バーだけをスマートに表示
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
    
    println("うまい変数変換の探索を開始します...")
    global hof_result = run_search(X_train, L_data, niterations=40)
    
    println("\n探索が完了しました！")
    display(hof_result) 
end