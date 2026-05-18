# main.jl
using SymbolicRegression
using DynamicExpressions # 微分機能(eval_grad_tree_array)を確実に使うため追加
using Zygote  

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
    local f_x
    try
        f_result = eval_tree_array(tree, dataset.X, options)
        f_x = f_result[1] # 1つ目は確実に計算値
        
        # フラグを使わず、計算結果に異常値(NaNやInf)が無いかを直接確認する
        if f_x === nothing || any(x -> !isfinite(x), f_x)
            return L(Inf)
        end
    catch
        return L(Inf)
    end
    
    # ② 微分 f'(x) の評価
    local df_dx
    try
        grad_result = eval_grad_tree_array(tree, dataset.X, options; variable=true)
        grads = grad_result[1] # 1つ目は確実に勾配(Gradient)
        
        if grads === nothing || any(x -> !isfinite(x), grads)
            return L(Inf)
        end
        df_dx = grads[1, :] # xに関する微分を取り出す
    catch
        return L(Inf)
    end
    
    v = dataset.X[2, :]
    L_target = dataset.y
    
    # ③ 変換後の作用
    L_pred_base = 0.5 .* (df_dx .* v).^2 .- 0.5 .* f_x.^2
    
    # ④ 定数のズレ(C)を自動で吸収して、誤差(MSE)を計算する
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
function run_search(X_train, L_data; niterations=40) # ← 探索回数を少し増やして定数を3.0まで合わせる時間を与えます
    options = Options(
        binary_operators=[+, -, *, /],# ^, # ← べき乗は今回の系ではカオスの原因になるので一旦封印！
        # unary_operators=[sin, cos, exp, sinh], # ← 今回の系ではカオスの原因になるので一旦封印！
        loss_function=action_loss,
        npopulations=30,
        parsimony=0.01,
        verbosity=1,      # ← これを0にすると、毎回の進行ログが消えて静かになります
        progress=true     # ← これをtrueにすると、下部にプログレスバーだけが表示されます
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
    global hof_result = run_search(X_train, L_data, niterations=10)
    
    println("探索が完了しました！")
    hof_result 
end