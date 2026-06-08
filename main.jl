# main.jl
using SymbolicRegression

# ==========================================
# ★追加: 多変数タスクを自動生成・命名するマクロ
# ==========================================
macro make_multi_task(num_in, num_out, lhs_expr, rhs_expr, x_range)
    lhs_str = string(lhs_expr)
    rhs_str = string(rhs_expr)
    name_str = "多変数変換探索 [ $lhs_str  ->  $rhs_str ]"
    
    return :(
        MultiTransformationTask(
            $name_str,
            [+, -, *, /],          # 使用する二項演算子
            [sq],                  # 使用する単項演算子
            $(esc(lhs_expr)),      # 左辺の関数
            $(esc(rhs_expr)),      # 右辺の関数
            $(esc(num_in)),        # 入力変数の数 (x, y...)
            $(esc(num_out)),       # 出力変数の数 (u, v...)
            $(esc(x_range))        # xの範囲
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
struct MultiTransformationTask
    name::String
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    lhs_function::Function  
    rhs_function::Function  
    num_inputs::Int
    num_outputs::Int
    x_range::Tuple{Float64, Float64} 
end

# ==========================================
# 2. 損失関数のジェネレータ (交互最適化対応)
# ==========================================
function make_loss_function(task::MultiTransformationTask, U_current::Matrix{Float64}, target_idx::Int)
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # ① 現在探索中の変数 (例: u) の予測値を計算
        f_res = eval_tree_array(tree, dataset.X, options)
        u_pred = f_res[1]

        if u_pred === nothing || any(x -> !isfinite(x), u_pred)
            return L(Inf)
        end
        
        # ② 他の変数 (例: v) は U_current の値で固定し、target_idx だけ差し替える
        U_temp = copy(U_current)
        U_temp[target_idx, :] .= u_pred
        
        # ③ RHSを評価 (U_temp の各行を引数として展開して渡す)
        RHS_pred_base = Float64[task.rhs_function(U_temp[:, i]...) for i in 1:size(U_temp, 2)]

        if any(x -> !isfinite(x), RHS_pred_base)
            return L(Inf)
        end

        LHS_target = dataset.y
        
        # ④ 定数ズレを吸収してMSE計算
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        RHS_pred = RHS_pred_base .+ C
        
        mse_loss = sum((RHS_pred .- LHS_target).^2) / length(LHS_target)

        return isnan(mse_loss) || isinf(mse_loss) ? L(Inf) : mse_loss
    end
    return custom_loss
end

# ==========================================
# 3. ホール・オブ・フェイム(HoF)から最良の木を抽出
# ==========================================
function get_best_tree(hof)
    best_loss = Inf
    best_tree = nothing
    for (i, member) in enumerate(hof.members)
        if hof.exists[i] && member.loss < best_loss
            best_loss = member.loss
            best_tree = member.tree
        end
    end
    return best_tree
end

# ==========================================
# 4. データ生成と交互探索(Coordinate Descent)の実行
# ==========================================
function run_multi_transformation_search(task::MultiTransformationTask; N::Int=200, niterations=20, max_rounds=2)
    println("【探索開始】 Task: ", task.name)
    
    # データの生成 (num_inputs行 × N列)
    low, high = task.x_range
    X_train = rand(task.num_inputs, N) .* (high - low) .+ low
    
    # 左辺を正解データ (ターゲット) として計算
    LHS_data = Float64[task.lhs_function(X_train[:, i]...) for i in 1:N]
    
    # 出力変数行列 U_current の初期化 (最初は入力変数をそのまま代入などの初期推測)
    U_current = zeros(task.num_outputs, N)
    for i in 1:min(task.num_inputs, task.num_outputs)
        U_current[i, :] .= X_train[i, :]
    end

    # 最終結果を保存する配列
    best_trees = Vector{Any}(undef, task.num_outputs)

    # 交互最適化のループ
    for round in 1:max_rounds
        println("\n--- ラウンド $round ---")
        for target_idx in 1:task.num_outputs
            var_name = target_idx == 1 ? "u" : (target_idx == 2 ? "v" : "var$target_idx")
            println("▶ 変数 [$var_name] を最適化中...")
            
            options = Options(
                binary_operators=task.binary_operators,
                unary_operators=task.unary_operators,
                loss_function=make_loss_function(task, U_current, target_idx),
                npopulations=30,
                parsimony=0.01,
                verbosity=0,  # ログが多すぎるのでミュート
                progress=true,
                early_stop_condition = (loss, complexity) -> loss < 1e-10 && complexity < 7
            )

            # 探索の実行
            hof = EquationSearch(X_train, LHS_data, options=options, niterations=niterations)
            
            # 最良の数式を抽出して U_current を更新
            best_tree = get_best_tree(hof)
            if best_tree !== nothing
                best_trees[target_idx] = best_tree
                f_res = eval_tree_array(best_tree, X_train, options)
                U_current[target_idx, :] .= f_res[1]
                
                # 見つかった数式を表示
                println("  見つかった $var_name の数式: ", string_tree(best_tree, options))
            else
                println("  有効な数式が見つかりませんでした。")
            end
        end
    end
    
    println("\n【交互探索完了！】")
    return best_trees
end

# ==========================================
# 5. 実行ブロック (テスト)
# ==========================================
if !@isdefined(IS_TESTING)
    # 2変数から2変数への平方完成テスト
    # LHS: 0.5(x^2 + 2x + y^2 + 4y)
    # RHS: 0.5(u^2 + v^2)
    # 期待される結果: u = x + 1, v = y + 2
    
    multi_task = @make_multi_task(
        2, 2,                                            # 入力2つ, 出力2つ
        (x, y) -> 0.5 * (x^2 + 2.0*x + y^2 + 4.0*y),     # LHS
        (u, v) -> 0.5 * (u^2 + v^2),                     # RHS
        (-5.0, 5.0)                                      # 範囲
    )

    run_multi_transformation_search(multi_task, max_rounds=2)

    nothing # 長い出力を消す
end