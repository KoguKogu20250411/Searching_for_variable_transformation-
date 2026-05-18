# test/runtests.jl
using Test
using SymbolicRegression

# 「今はテスト中である」というフラグを定義しておく
const IS_TESTING = true

# 親ディレクトリにある main.jl を読み込む（上のフラグのおかげで探索は自動実行されない）
include(joinpath(@__DIR__, "..", "main.jl"))

@testset "Quadraticizability Variable Transformation" begin
    
    @testset "1. Data Generation Test" begin
        # N=50で正しく行列とベクトルが生成されるか
        X, L = generate_data(50)
        @test size(X) == (2, 50)
        @test length(L) == 50
        @test typeof(X) <: Matrix{Float64}
    end

    @testset "2. Action Loss Function Safety Test" begin
        X, L = generate_data(10)
        dataset = Dataset(X, L)
        options = Options(binary_operators=[+, -])
        
        # SymbolicRegressionのダミー数式ノードを作成（ f(x) = x_1 の意味 ）
        tree = Node(Float64; feature=1)
        
        # Loss関数がエラー落ちせずに数値を返すかを検証
        loss = action_loss(tree, dataset, options)
        @test typeof(loss) <: Float64
        @test !isnan(loss)
        @test loss != Inf
    end

    @testset "3. Equation Search Execution Test" begin
        X, L = generate_data(10)
        # イテレーションを極小（1回）にして、アルゴリズム全体が正常に稼働して結果を返すかを検証
        hof = run_search(X, L, niterations=1)
        @test hof !== nothing
    end
    
end
nothing # test実行時に標準出力に大量のメッセージが出ないようにするおまじない