using SymbolicRegression

# --- 準備: 物理で使う独自の関数を定義 ---
sq(x) = x^2  # 二乗関数

# ==========================================
# 1. データセットの作成 (真の物理系)
# ==========================================
println("データを生成しています...")
N = 200
x_data = rand(N) .* 10.0 .- 5.0
v_data = rand(N) .* 10.0 .- 5.0

# 観測されたラグランジアン: L = 1/2 v^2 - (1/2 x^2 - 3x)
L_data = 0.5 .* v_data.^2 .- (0.5 .* x_data.^2 .- 3.0 .* x_data)

# 特徴量行列を作成（1行目がx, 2行目がv）
# 修正点: ' (Adjoint) ではなく、明示的に通常の行列(Matrix)に変換します
X_train = Matrix(hcat(x_data, v_data)') 

# ==========================================
# 2. カスタム目的関数 (Loss) の定義
# ==========================================
function action_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    
    # ① 候補の数式 f(x) にデータ x を入れて計算する
    # 修正点: eval_tree_array は最新版では (値の配列, フラグ) を返しますが、
    # 失敗した場合はエラーをキャッチする方が確実です。
    local f_x, grads
    try
        # 評価を実行 (ここでは f_x と flag のタプルが返る想定ですが、仕様変更に備えて全体を取得)
        eval_result = eval_tree_array(tree, dataset.X, options)
        
        # 成功したかどうかの判定 (フラグが配列の場合は all を使う、あるいは単一のBoolならそのまま)
        if typeof(eval_result[2]) <: AbstractArray
            if !all(eval_result[2])
                return L(Inf)
            end
        elseif !eval_result[2]
             return L(Inf)
        end
        f_x = eval_result[1]
        
    catch e
        return L(Inf) # ゼロ割りなどのエラーは無視する
    end
    
    # ② 数式の微分 f'(x) を計算する
    try
        # gradient を取得
        grad_result = eval_grad_tree_array(tree, dataset.X, options; variable=true)
        
        if typeof(grad_result[2]) <: AbstractArray
            if !all(grad_result[2])
                return L(Inf)
            end
        elseif !grad_result[2]
             return L(Inf)
        end
        grads = grad_result[1]
    catch e
        return L(Inf)
    end
    
    df_dx = grads[1, :] # 1つ目の変数 (x) に関する微分を抽出
    
    v = dataset.X[2, :] # 速度データ
    L_target = dataset.y  # 正解のラグランジアンデータ
    
    # ③ 候補の変数変換によって予測されるラグランジアン
    L_pred_base = 0.5 .* (df_dx .* v).^2 .- 0.5 .* f_x.^2
    
    # ④ 定数のズレ(C)を自動で吸収して、誤差(MSE)を計算する
    C = sum(L_target .- L_pred_base) / length(L_target)
    L_pred = L_pred_base .+ C
    
    mse_loss = sum((L_pred .- L_target).^2) / length(L_target)
    
    # NaNやInfが含まれている場合はInfを返す（安全対策）
    if isnan(mse_loss) || isinf(mse_loss)
        return L(Inf)
    end
    
    return mse_loss
end

# ==========================================
# 3. 探索の設定と実行
# ==========================================
println("うまい変数変換の探索を開始します...")

options = Options(
    binary_operators=[+, -, *, /],
    unary_operators=[sq], 
    loss_function=action_loss, 
    npopulations=30, 
    parsimony=0.01   
)

# EquationSearchの実行
hof = EquationSearch(X_train, L_data, options=options, niterations=10)

println("探索が完了しました！結果を確認してください。")