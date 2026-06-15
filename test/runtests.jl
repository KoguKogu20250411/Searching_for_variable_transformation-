using Test
using SymbolicRegression

# main.jl 末尾のテスト実行ブロックが走らないようにフラグを立てる
const IS_TESTING = true

# テスト対象のファイルを読み込み (ディレクトリ構成に合わせてパスは適宜変更してください)
include(joinpath(@__DIR__, "main.jl"))

@testset "SymbolicRegression Transformation Framework Tests" begin
    
    @testset "0. Utility Functions" begin
        # 共通ユーティリティ関数の動作確認
        @test sq(4.0) == 16.0
        @test sq(-3.0) == 9.0
        
        @test safe_sqrt(9.0) == 3.0
        @test safe_sqrt(-9.0) == 3.0 # abs(x) が正常に適用されているか
    end

    @testset "1. MultiTransformationTask Definition" begin
        # 多変数タスク構造体が正しく初期化され、クロージャが機能するか
        multi_task = MultiTransformationTask(
            "Test Multi Task",
            [+, -],
            [sq],
            (x, y) -> 2.0 * x + y,       # LHS
            (u, v) -> u^2 + v^2,         # RHS
            2, 2, (-2.0, 2.0)
        )
        
        @test multi_task.name == "Test Multi Task"
        @test length(multi_task.binary_operators) == 2
        @test multi_task.num_inputs == 2
        @test multi_task.num_outputs == 2
        @test multi_task.lhs_function(3.0, 4.0) == 10.0
        @test multi_task.rhs_function(2.0, 3.0) == 13.0
        @test multi_task.x_range == (-2.0, 2.0)
    end

    @testset "2. LagrangianTask Definition" begin
        # ラグランジアンタスク構造体が正しく初期化され、クロージャが機能するか
        lag_task = LagrangianTask(
            "Test Lagrangian Task",
            [+, -, *, /],
            [sq],
            (x, v) -> 0.5 * v^2 - 0.5 * x^2,             # LHS
            (u, u_dot) -> 0.5 * u_dot^2 - 0.5 * u^2,     # RHS
            (-5.0, 5.0), (-5.0, 5.0)
        )
        
        @test lag_task.name == "Test Lagrangian Task"
        @test length(lag_task.binary_operators) == 4
        @test lag_task.lhs_function(2.0, 3.0) == 0.5 * 3.0^2 - 0.5 * 2.0^2
        @test lag_task.rhs_function(1.0, 2.0) == 0.5 * 2.0^2 - 0.5 * 1.0^2
        @test lag_task.x_range == (-5.0, 5.0)
        @test lag_task.v_range == (-5.0, 5.0)
    end

    @testset "3. Execution Pipeline (Smoke Tests)" begin
        # CI/CD等で長時間ブロックされるのを防ぐため、
        # サンプル数(N)とイテレーション数(niterations)を極小にして「エラー落ちしないか」を担保する

        @testset "MultiTransformation Search" begin
            test_multi_task = MultiTransformationTask(
                "Smoke Test Multi",
                [+, -], [sq],
                (x, y) -> 0.5 * (x^2 + y^2),
                (u, v) -> 0.5 * (u^2 + v^2),
                2, 2, (-5.0, 5.0)
            )
            
            # max_rounds=1 にしてテストを高速化
            best_trees = run_multi_transformation_search(test_multi_task; N=10, niterations=1, max_rounds=1)
            
            # 探索結果が配列として返ってきているか、要素数が num_outputs と一致するか
            @test best_trees isa AbstractVector
            @test length(best_trees) == 2
        end

        @testset "Lagrangian Search" begin
            test_lag_task = LagrangianTask(
                "Smoke Test Lagrangian",
                [+, -], [sq],
                (x, v) -> 0.5 * v^2 - 0.5 * x^2,
                (u, u_dot) -> 0.5 * u_dot^2 - 0.5 * u^2,
                (-5.0, 5.0), (-5.0, 5.0)
            )
            
            hof = run_lagrangian_search(test_lag_task; N=10, niterations=1)
            
            # 探索結果（HallOfFameオブジェクト）が正常に生成されて返ってきているか
            @test hof !== nothing
        end
    end

end