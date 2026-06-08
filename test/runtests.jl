using Test
using SymbolicRegression

const IS_TESTING = true
include(joinpath(@__DIR__, "..", "main.jl"))

@testset "Transformation Framework Tests" begin
    
    @testset "Utility Functions" begin
        # 追加した基底関数が正しく動作するかテスト
        @test sq(4.0) == 16.0
        @test sq(-3.0) == 9.0
        
        @test safe_sqrt(9.0) == 3.0
        @test safe_sqrt(-9.0) == 3.0 # abs(x)が正常に適用されているか確認
    end

    @testset "TransformationTask Definition" begin
        # タスク構造体が正しく初期化され、関数が評価できるかテスト
        task = TransformationTask(
            "Mock Task",
            [+, -],
            [sq],
            x -> 2.0 * x,
            u -> u^2,
            (-2.0, 2.0)
        )
        
        @test task.name == "Mock Task"
        @test length(task.binary_operators) == 2
        @test task.lhs_function(3.0) == 6.0
        @test task.rhs_function(4.0) == 16.0
        @test task.x_range == (-2.0, 2.0)
    end

    @testset "Integration & Execution Pipeline" begin
        # 平方完成のタスクを使って、探索全体がクラッシュせずに動くかを結合テスト
        square_completion_task = TransformationTask(
            "Square Completion Test",
            [+, -, *, /],
            [sq],
            x -> 0.5 * x^2 - 3.0 * x,
            u -> 0.5 * u^2,
            (-5.0, 5.0)
        )
        
        # ユニットテスト（CI/CD等）で長時間ブロックされるのを防ぐため、
        # サンプル数(N)とイテレーション数(niterations)を極小にして「エラー落ちしないか」を担保する
        hof = run_transformation_search(square_completion_task; N=20, niterations=1)
        
        # 探索結果（HallOfFameオブジェクト）が正常に生成されて返ってきているか
        @test hof !== nothing
    end

end