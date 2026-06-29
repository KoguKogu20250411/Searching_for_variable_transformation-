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
    prev_tree_strings = fill("", task.num_outputs)

    for round in 1:max_rounds
        println("\n--- ラウンド $round ---")
        changed_in_this_round = false
        current_maxsize = min(5 + (round - 1) * 3, 15)
        current_parsimony = max(0.5 * (0.2 ^ (round - 1)), 0.001)
        
        for target_idx in 1:task.num_outputs
            var_name = target_idx == 1 ? "u" : (target_idx == 2 ? "v" : "var$target_idx")
            
            # 修正: verbosity=0 で progress=true を使用できない問題を解決するため verbosity=1 に変更
            options = Options(
                binary_operators=task.binary_operators,
                unary_operators=task.unary_operators,
                loss_function=make_loss_function(task, U_current, target_idx),
                npopulations=30,
                parsimony=current_parsimony,    
                maxsize=current_maxsize,       
                verbosity=1,  
                progress=true,
                early_stop_condition = (loss, complexity) -> loss < 1e-5 && complexity < 10
            )

            hof = EquationSearch(X_train, LHS_data, options=options, niterations=niterations)
            best_tree = get_best_tree(hof)
            if best_tree !== nothing
                best_trees[target_idx] = best_tree
                U_current[target_idx, :] .= eval_tree_array(best_tree, X_train, options)[1]
                tree_str = string_tree(best_tree, options)
                if tree_str != prev_tree_strings[target_idx]
                    changed_in_this_round = true
                    prev_tree_strings[target_idx] = tree_str
                end
            end
        end
        if !changed_in_this_round && round > 1 break end
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

function make_loss_function(task::LagrangianTask, V_vel::Matrix{Float64})
    function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        f_res = eval_tree_array(tree, dataset.X, options)
        u_pred = f_res[1]
        if u_pred === nothing || any(x -> !isfinite(x), u_pred) return L(Inf) end
        
        dx = 1e-4
        X_plus = copy(dataset.X) .+ dx
        X_minus = copy(dataset.X) .- dx
        df_dx = (eval_tree_array(tree, X_plus, options)[1] .- eval_tree_array(tree, X_minus, options)[1]) ./ (2.0 * dx)
        u_dot_pred = df_dx .* V_vel[1, :]
        RHS_pred_base = task.rhs_function.(u_pred, u_dot_pred)
        
        LHS_target = dataset.y
        C = sum(LHS_target .- RHS_pred_base) / length(LHS_target)
        return sum((RHS_pred_base .+ C .- LHS_target).^2) / length(LHS_target)
    end
    return custom_loss
end

function run_lagrangian_search(task::LagrangianTask; N::Int=200, niterations=40)
    low_x, high_x = task.x_range
    low_v, high_v = task.v_range
    X_pos = rand(1, N) .* (high_x - low_x) .+ low_x
    V_vel = rand(1, N) .* (high_v - low_v) .+ low_v
    LHS_data = Float64[task.lhs_function(X_pos[1, i], V_vel[1, i]) for i in 1:N]
    
    options = Options(binary_operators=task.binary_operators, unary_operators=task.unary_operators,
                      loss_function=make_loss_function(task, V_vel), npopulations=30,
                      verbosity=1, progress=true)

    hof = EquationSearch(X_pos, LHS_data, options=options, niterations=niterations)
    best_tree = get_best_tree(hof)
    println("【ラグランジアン探索完了！】 見つかった数式: ", string_tree(best_tree, options))
    return hof
end


# ==========================================
# 3. テスト実行ブロック
# ==========================================
if !@isdefined(IS_TESTING)
    # ----------------------------------------------------
    # テストA: 1変数の最適化
    # ----------------------------------------------------
    # multi_task_A = @make_multi_task(1, 1, (x) -> 0.5*(x^2 + 2x + 1), (u) -> 0.5*u^2, (-5.0, 5.0))
    # run_multi_transformation_search(multi_task_A, max_rounds=2)

    # ----------------------------------------------------
    # テストB: 2変数・多変数の交互最適化
    # ----------------------------------------------------
    multi_task_B = @make_multi_task(2, 2, (x, y) -> 0.5*(x^2 + 2x + y^2 + 4y + 5), (u, v) -> 0.5*(u^2 + v^2), (-5.0, 5.0))
    run_multi_transformation_search(multi_task_B, max_rounds=3)

    # ----------------------------------------------------
    # テストC: ラグランジアン変換
    # ----------------------------------------------------
    # lagrangian_task = @make_lagrangian_task((x, v) -> 0.5*v^2 - 0.5*(x - 3)^2, (u, u_dot) -> 0.5*u_dot^2 - 0.5*u^2, (-5.0, 5.0), (-5.0, 5.0))
    # run_lagrangian_search(lagrangian_task)
end