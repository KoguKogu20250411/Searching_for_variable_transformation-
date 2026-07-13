# main.jl
using SymbolicRegression

# ==========================================
# 0. 共通ユーティリティ・基底関数
# ==========================================
sq(x) = x^2
safe_sqrt(x) = sqrt(abs(x))

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
# 1. 【多変数 ↔ 多変数】 交互最適化タスク
# ==========================================
macro make_multi_task(num_in, num_out, lhs_expr, rhs_expr, x_range)
    name_str = "多変数変換探索 [ $(string(lhs_expr))  ->  $(string(rhs_expr)) ]"
    return :(
        MultiTransformationTask(
            $name_str,
            [+, -, *, /], [sq],
            $(esc(lhs_expr)), $(esc(rhs_expr)),
            $(esc(num_in)), $(esc(num_out)), $(esc(x_range))
        )
    )
end

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

function make_loss_function(task::MultiTransformationTask, U_current::Matrix{Float64}, target_idx::Int)
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        f_res = eval_tree_array(tree, dataset.X, options)
        u_pred = f_res[1]

        if u_pred === nothing || any(x -> !isfinite(x), u_pred) return L(Inf) end
        
        U_temp = copy(U_current)
        U_temp[target_idx, :] .= u_pred
        
        RHS_pred_base = Float64[task.rhs_function(U_temp[:, i]...) for i in 1:size(U_temp, 2)]
        if any(x -> !isfinite(x), RHS_pred_base) return L(Inf) end

        LHS_target = dataset.y
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        RHS_pred = RHS_pred_base .+ C
        
        mse_loss = sum((RHS_pred .- LHS_target).^2) / length(LHS_target)
        return isnan(mse_loss) || isinf(mse_loss) ? L(Inf) : mse_loss
    end
    return custom_loss
end

function run_multi_transformation_search(task::MultiTransformationTask; N::Int=200, niterations=20, max_rounds=2)
    println("\n=========================================")
    println("【探索開始】 ", task.name)
    println("=========================================")
    
    low, high = task.x_range
    X_train = rand(task.num_inputs, N) .* (high - low) .+ low
    LHS_data = Float64[task.lhs_function(X_train[:, i]...) for i in 1:N]
    
    U_current = zeros(task.num_outputs, N)
    for i in 1:min(task.num_inputs, task.num_outputs)
        U_current[i, :] .= X_train[i, :]
    end

    best_trees = Vector{Any}(undef, task.num_outputs)
    prev_tree_strings = fill("", task.num_outputs) # 収束判定用の文字列保存配列

    for round in 1:max_rounds
        println("\n--- ラウンド $round ---")
        changed_in_this_round = false # 収束フラグ

        for target_idx in 1:task.num_outputs
            var_name = target_idx == 1 ? "u" : (target_idx == 2 ? "v" : "var$target_idx")
            println("▶ 変数 [$var_name] を最適化中...")
            
            options = Options(
                binary_operators=task.binary_operators,
                unary_operators=task.unary_operators,
                loss_function=make_loss_function(task, U_current, target_idx),
                npopulations=40,              # 多様性確保のため少し増やす
                parsimony=0.1,                # 【改善】複雑な式へのペナルティを強くする
                maxsize=15,                   # 【改善】式の最大サイズを制限して過学習を防ぐ
                verbosity=1,
                progress=true,
                early_stop_condition = (loss, complexity) -> loss < 1e-5 && complexity < 10 # 【改善】終了条件を少し緩和
            )

            hof = EquationSearch(X_train, LHS_data, options=options, niterations=niterations)
            
            best_tree = get_best_tree(hof)
            if best_tree !== nothing
                best_trees[target_idx] = best_tree
                f_res = eval_tree_array(best_tree, X_train, options)
                U_current[target_idx, :] .= f_res[1]
                
                tree_str = string_tree(best_tree, options)
                println("  見つかった $var_name の数式: ", tree_str)
                
                # 前回見つかった数式と異なるかチェック
                if tree_str != prev_tree_strings[target_idx]
                    changed_in_this_round = true
                    prev_tree_strings[target_idx] = tree_str
                end
            else
                println("  有効な数式が見つかりませんでした。")
            end
        end

        # 収束判定: 前回からどの数式も変化していなければ探索を打ち切る
        if !changed_in_this_round && round > 1
            println("\n【早期終了】 前回のラウンドから数式に変化がありません。収束したとみなします。")
            break
        end
    end
    println("【交互探索完了！】")
    return best_trees
end


# ==========================================
# 2. 【ラグランジアン】 連鎖律（チェーンルール）タスク
# ==========================================
macro make_lagrangian_task(lhs_expr, rhs_expr, x_range, v_range)
    name_str = "ラグランジアン変換 [ $(string(lhs_expr))  ->  $(string(rhs_expr)) ]"
    return :(
        LagrangianTask(
            $name_str,
            [+, -, *, /], [sq],
            $(esc(lhs_expr)), $(esc(rhs_expr)),
            $(esc(x_range)), $(esc(v_range))
        )
    )
end

struct LagrangianTask
    name::String
    binary_operators::Vector{Function}
    unary_operators::Vector{Function}
    lhs_function::Function  
    rhs_function::Function  
    x_range::Tuple{Float64, Float64}
    v_range::Tuple{Float64, Float64}
end

# 修正箇所: AIは x(位置) しか知らない。速度 v はクロージャ(外部引数)として直接渡す。
function make_loss_function(task::LagrangianTask, V_vel::Matrix{Float64})
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        # dataset.X には x のデータ (1次元) しか入っていない
        f_res = eval_tree_array(tree, dataset.X, options)
        u_pred = f_res[1]
        if u_pred === nothing || any(x -> !isfinite(x), u_pred) return L(Inf) end
        
        dx = 1e-4
        
        X_plus = copy(dataset.X)
        X_plus[1, :] .+= dx
        
        X_minus = copy(dataset.X)
        X_minus[1, :] .-= dx

        u_plus = eval_tree_array(tree, X_plus, options)[1]
        u_minus = eval_tree_array(tree, X_minus, options)[1]
        if u_plus === nothing || u_minus === nothing return L(Inf) end

        df_dx = (u_plus .- u_minus) ./ (2.0 * dx)
        
        # 連鎖律 du/dt = (du/dx) * v (vは外部から持ち込んだ V_vel を使用)
        u_dot_pred = df_dx .* V_vel[1, :]
        if any(x -> !isfinite(x), u_dot_pred) return L(Inf) end

        RHS_pred_base = task.rhs_function.(u_pred, u_dot_pred)
        if any(x -> !isfinite(x), RHS_pred_base) return L(Inf) end

        LHS_target = dataset.y
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        RHS_pred = RHS_pred_base .+ C
        
        mse_loss = sum((RHS_pred .- LHS_target).^2) / length(LHS_target)
        return isnan(mse_loss) || isinf(mse_loss) ? L(Inf) : mse_loss
    end
    return custom_loss
end

function run_lagrangian_search(task::LagrangianTask; N::Int=200, niterations=40)
    println("\n=========================================")
    println("【探索開始】 ", task.name)
    println("=========================================")
    
    low_x, high_x = task.x_range
    low_v, high_v = task.v_range
    X_pos = rand(1, N) .* (high_x - low_x) .+ low_x
    V_vel = rand(1, N) .* (high_v - low_v) .+ low_v
    
    # ターゲット値 (LHS_data) は x と v の両方から計算する
    LHS_data = Float64[task.lhs_function(X_pos[1, i], V_vel[1, i]) for i in 1:N]
    
    options = Options(
        binary_operators=task.binary_operators,
        unary_operators=task.unary_operators,
        loss_function=make_loss_function(task, V_vel), # ここで速度を直接渡す！
        npopulations=30, parsimony=0.01,
        verbosity=1,
        progress=true,
        early_stop_condition = (loss, complexity) -> loss < 1e-10 && complexity < 7
    )

    # 修正箇所: AIには X_pos(位置) しか渡さない。これで確実に u(x) を探索する。
    hof = EquationSearch(X_pos, LHS_data, options=options, niterations=niterations)
    
    println("\n【ラグランジアン探索完了！】")
    best_tree = get_best_tree(hof)
    if best_tree !== nothing
        println("  見つかった座標変換 u(x) の数式: ", string_tree(best_tree, options))
    end
    return hof
end


# ==========================================
# 3. 実行・テストブロック
# ==========================================
if !@isdefined(IS_TESTING)
    
        # ----------------------------------------------------
    # テストA: 1変数の最適化 (Coordinate Descent)
    # ----------------------------------------------------
    # multi_task = @make_multi_task(
    #     1, 1,                                            # 入力1つ, 出力1つ
    #     (x) -> 0.5 * (x^2 + 2.0*x + 1.0),                # LHS: 平方完成前
    #     (u) -> 0.5 * (u^2),                               # RHS: 平方完成後
    #     (-5.0, 5.0)                                      # 範囲
    # )
    
    # # max_rounds=1 で uを探索します
    # run_multi_transformation_search(multi_task, max_rounds=1, niterations=20)


    # ----------------------------------------------------
    # テストB: 2変数・多変数の交互最適化 (Coordinate Descent)
    # ----------------------------------------------------
    # multi_task = @make_multi_task(
    #     2, 2,                                            # 入力2つ, 出力2つ
    #     (x, y) -> 0.5 * (x^2 + 2.0*x + y^2 + 4.0*y + 5.0),     # LHS: 平方完成前
    #     (u, v) -> 0.5 * (u^2 + v^2),                     # RHS: 平方完成後
    #     (-5.0, 5.0)                                      # 範囲
    # )
    
    # # max_rounds=2 で u と v を2回ずつ交互に探索します
    # run_multi_transformation_search(multi_task, max_rounds=2, niterations=20)


    # ----------------------------------------------------
    # テストC: ラグランジアン変換 (微分の連鎖律)
    # ----------------------------------------------------
    lagrangian_task = @make_lagrangian_task(
        (x, v) -> 0.5 * v^2 - 0.5 * (x - 3.0)^2,         # LHS: 原点がズレた調和振動子
        (u, u_dot) -> 0.5 * u_dot^2 - 0.5 * u^2,         # RHS: 綺麗な調和振動子
        (-5.0, 5.0),                                     # 位置 x の範囲
        (-5.0, 5.0)                                      # 速度 v の範囲
    )

    run_lagrangian_search(lagrangian_task, niterations=40)

    # ----------------------------------------------------
    nothing # スクリプト末尾の不要な長文出力をミュート
end