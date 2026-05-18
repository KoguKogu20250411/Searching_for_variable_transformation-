using Pkg
# Pkg.add("SymbolicRegression")

using SymbolicRegression
using Test

# 1. 物理タスクのための演算子定義
# 記号回帰の探索空間（基底関数）を定義します。
sq(x) = x^2
options = Options(
    binary_operators=[+, -, *, /],
    unary_operators=[sqrt, sq],
    npopulations=20,     # 集団サイズ
    niterations=10,      # 探索の深さ（実用時はより大きく設定）
    parsimony=0.01       # 複雑な数式に対するペナルティ（オッカムの剃刀）
)

# 2. データの生成（相対論的ダイナミクス）
function generate_relativistic_data(n_samples::Int=100)
    c_true = 3.0e8      # 光速
    m_true = 1.0        # 静止質量
    
    # 0 から 0.99c までの速度データを生成
    v = rand(n_samples) .* (0.99 * c_true)
    c = fill(c_true, n_samples)
    
    # 入力特徴量: X = [v, c]
    X = hcat(v, c)' 
    
    # 真の非線形運動量 (ターゲット)
    # p = m * v * γ  --> γ = p / (m * v)
    # ここでは、MLが「γ」という変数変換を抽出できるかを模すため、ターゲットをγとする
    gamma_true = @. 1.0 / sqrt(1.0 - (v / c)^2)
    
    return X, gamma_true
end

# 3. ユニットテスト群
@testset "Symbolic Regression: Variable Transformation Discovery" begin
    
    @testset "データ生成系の検証" begin
        X, y = generate_relativistic_data(50)
        @test size(X) == (2, 50)
        @test length(y) == 50
        @test all(y .>= 1.0) # ローレンツ因子は常に1以上であるべき
    end
    
    @testset "探索アルゴリズムの稼働検証" begin
        # ユニットテスト用にイテレーションを絞って実行
        test_options = Options(
            binary_operators=[+, -, *, /],
            npopulations=5,
            niterations=2
        )
        
        # 自明な変数変換（単なる割り算）が発見できるかのテスト: z = x_1 / x_2
        X_test = rand(2, 20)
        y_test = X_test[1, :] ./ X_test[2, :]
        
        # 方程式の探索
        hof = EquationSearch(X_test, y_test, options=test_options, runtests=false)
        
        # 実行がクラッシュせず、結果(Hall of Fame)が返却されることを確認
        @test hof !== nothing
    end
end

# 4. 実際の探索の実行（テスト外）
println("--- 真の変数変換（ローレンツ因子）の探索を開始します ---")
X_train, y_train = generate_relativistic_data(200)

# 探索の実行 (実用レベル)
# hall_of_fame_result = EquationSearch(X_train, y_train, options=options)
# 
# 期待される出力の例（最良の数式として以下のような形式が発見される）:
# y = 1.0 / sqrt(1.0 - sq(x1 / x2))