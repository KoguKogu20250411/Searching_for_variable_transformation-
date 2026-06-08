# main.jl
using SymbolicRegression

# ==========================================
# ★追加: ラグランジアンタスク用マクロ
# ==========================================
macro make_lagrangian_task(lhs_expr, rhs_expr, x_range, v_range)
    lhs_str = string(lhs_expr)
    rhs_str = string(rhs_expr)
    name_str = "ラグランジアン変換 [ $lhs_str  ->  $rhs_str ]"
    
    return :(
        LagrangianTask(
            $name_str,
            [+, -, *, /],          # 使用する二項演算子
            [sq],                  # 使用する単項演算子
            $(esc(lhs_expr)),      # 左辺の関数: L(x, v)
            $(esc(rhs_expr)),      # 右辺の関数: L(u, u_dot)
            $(esc(x_range)),       # 位置 x のサンプリング範囲
            $(esc(v_range))        # 速度 v のサンプリング範囲
        )
    )
end

# ==========================================
# 0. ユーザー定義の基底関数
# ==========================================
sq(x) = x^2
safe_sqrt(x) = sqrt(abs(x))

# ==========================================
# 1. 探索タスクの定義
# ==========================================
struct LagrangianTask
    name::String
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    lhs_function::Function  
    rhs_function::Function  
    x_range::Tuple{Float64, Float64}
    v_range::Tuple{Float64, Float64}
end

# ==========================================
# 2. 損失関数のジェネレータ (微分の連鎖律を実装)
# ==========================================
function make_loss_function(task::LagrangianTask)
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        
        # dataset.X は 2行N列 の行列。(1行目: 位置x, 2行目: 速度v)
        # eval_tree_array には x だけを渡すため 1行N列 の行列として切り出す
        X_pos = dataset.X[1:1, :] 
        V_vel = dataset.X[2, :]   # 速度はベクトルとして保持

        # ① 新しい座標 u = f(x) の評価
        f_res = eval_tree_array(tree, X_pos, options)
        u_pred = f_res[1]

        if u_pred === nothing || any(x -> !isfinite(x), u_pred)
            return L(Inf)
        end
        
        # ② 数値微分(中心差分)で df/dx を計算
        dx = 1e-4
        u_plus = eval_tree_array(tree, X_pos .+ dx, options)[1]
        u_minus = eval_tree_array(tree, X_pos .- dx, options)[1]

        if u_plus === nothing || u_minus === nothing
            return L(Inf)
        end

        df_dx = (u_plus .- u_minus) ./ (2.0 * dx)

        # ③ 連鎖律: 新しい速度 u_dot = (df/dx) * v を計算！
        u_dot_pred = df_dx .* V_vel

        if any(x -> !isfinite(x), u_dot_pred)
            return L(Inf)
        end

        # ④ 変換後のラグランジアンを評価: RHS(u, u_dot)
        RHS_pred_base = task.rhs_function.(u_pred, u_dot_pred)

        if any(x -> !isfinite(x), RHS_pred_base)
            return L(Inf)
        end

        LHS_target = dataset.y
        
        # ⑤ 定数ズレ(C)を吸収してMSEを計算
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        RHS_pred = RHS_pred_base .+ C
        
        mse_loss = sum((RHS_pred .- LHS_target).^2) / length(LHS_target)

        return isnan(mse_loss) || isinf(mse_loss) ? L(Inf) : mse_loss
    end

    return custom_loss
end

# ==========================================
# 3. データ生成と探索の実行
# ==========================================
function run_lagrangian_search(task::LagrangianTask; N::Int=200, niterations=40)
    println("【探索開始】 Task: ", task.name)
    
    # 位置 x と 速度 v のデータを個別にサンプリング
    low_x, high_x = task.x_range
    low_v, high_v = task.v_range
    
    X_pos = rand(1, N) .* (high_x - low_x) .+ low_x
    V_vel = rand(1, N) .* (high_v - low_v) .+ low_v
    
    # 2行N列の学習データ (1行目がx, 2行目がv)
    X_train = vcat(X_pos, V_vel)
    
    # 左辺を正解データ (LHSターゲット) として計算
    LHS_data = Float64[task.lhs_function(X_train[1, i], X_train[2, i]) for i in 1:N]
    
    options = Options(
        binary_operators=task.binary_operators,
        unary_operators=task.unary_operators,
        loss_function=make_loss_function(task),
        npopulations=30,
        parsimony=0.01,
        verbosity=1,
        progress=true,
        early_stop_condition = (loss, complexity) -> loss < 1e-10 && complexity < 7
    )

    hof = EquationSearch(X_train, LHS_data, options=options, niterations=niterations)
    
    println("\n【探索完了】")
    return hof
end

# ==========================================
# 4. 実行ブロック (テスト)
# ==========================================
if !@isdefined(IS_TESTING)
    # 【テスト問題】: 原点が x=3 にズレた調和振動子
    # 変換前 LHS: L = 0.5*v^2 - 0.5*(x - 3)^2
    # 変換後 RHS: L = 0.5*u_dot^2 - 0.5*u^2  (綺麗な調和振動子)
    # 期待される正解変換: u = x - 3
    
    lagrangian_task = @make_lagrangian_task(
        (x, v) -> 0.5 * v^2 - 0.5 * (x - 3.0)^2,  # 左辺 L(x, v)
        (u, u_dot) -> 0.5 * u_dot^2 - 0.5 * u^2,  # 右辺 L(u, u_dot)
        (-5.0, 5.0),                              # 位置 x の範囲
        (-5.0, 5.0)                               # 速度 v の範囲
    )

    run_lagrangian_search(lagrangian_task)

    nothing # 長い出力を消す
end