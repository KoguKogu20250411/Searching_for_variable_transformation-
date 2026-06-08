# main.jl
using SymbolicRegression

# ==========================================
# 0. ユーザー定義の基底関数（必要に応じて追加）
# ==========================================
sq(x) = x^2
safe_sqrt(x) = sqrt(abs(x))
# 例: gaussian(x) = exp(-x^2) など

# ==========================================
# 1. 探索タスクの定義
# ==========================================
struct TransformationTask
    name::String
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    lhs_function::Function  # 変換前の関数 (Source: xを入れたらLHSの値を返す)
    rhs_function::Function  # 変換後の関数 (Target: uを入れたらRHSの値を返す)
    x_range::Tuple{Float64, Float64} # 探索する x の範囲
end

# ==========================================
# 2. 損失関数のジェネレータ
# ==========================================
# タスクを受け取り、SymbolicRegression専用の loss_function を動的に生成します
function make_loss_function(task::TransformationTask)
    
    # 内部で明示的に名前付き関数を定義する（構文エラー回避のため）
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # ① 候補となる変数変換 u = f(x) の評価
        f_res = eval_tree_array(tree, dataset.X, options)
        u_pred = f_res[1]

        if u_pred === nothing || any(x -> !isfinite(x), u_pred)
            return L(Inf)
        end
        
        # ② 右辺の関数に変換後の変数 u を代入して評価: RHS(u)
        RHS_pred_base = task.rhs_function.(u_pred)

        if any(x -> !isfinite(x), RHS_pred_base)
            return L(Inf)
        end

        LHS_target = dataset.y
        
        # ③ 定数のズレ(C)を自動で吸収して誤差を計算
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        RHS_pred = RHS_pred_base .+ C
        
        # ④ MSEの計算
        mse_loss = sum((RHS_pred .- LHS_target).^2) / length(LHS_target)

        return isnan(mse_loss) || isinf(mse_loss) ? L(Inf) : mse_loss
    end

    # 生成した関数（クロージャ）を返す
    return custom_loss
end

# ==========================================
# 3. データ生成と探索の実行
# ==========================================
function run_transformation_search(task::TransformationTask; N::Int=200, niterations=40)
    println("【探索開始】 Task: ", task.name)
    
    # データの生成
    low, high = task.x_range
    x_data = rand(N) .* (high - low) .+ low
    
    # 左辺を正解データ (ターゲット) として計算
    LHS_data = task.lhs_function.(x_data)
    X_train = reshape(x_data, 1, N) # 1行N列のマトリックス
    
    # オプションの設定
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

    # 探索の実行
    hof = EquationSearch(X_train, LHS_data, options=options, niterations=niterations)
    
    println("\n【探索完了】")
    #display(hof)
    return hof
end

# ==========================================
# 4. 実行ブロック (テスト)
# ==========================================
if !@isdefined(IS_TESTING)
    # 例: 前回の「平方完成」の問題を汎用フレームワークで解いてみる
    # LHS: 0.5*x^2 - 3.0*x
    # RHS: 0.5*u^2
    # 期待される変換: u = x - 3
    
    square_completion_task = TransformationTask(
        "平方完成の探索 (x^2 - 3x -> u^2)",
        [+, -, *, /],          # 使用する二項演算子
        [sq],                  # 使用する単項演算子 (基底関数)
        x -> 0.5 * x^2 - 3.0 * x, # 左辺 LHS(x)
        u -> 0.5 * u^2,           # 右辺 RHS(u)
        (-5.0, 5.0)            # xの範囲
    )

    run_transformation_search(square_completion_task)

    nothing # 長い出力を消す
end