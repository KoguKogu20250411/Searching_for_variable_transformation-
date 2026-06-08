using Test

const IS_TESTING = true

include(joinpath(@__DIR__, "..", "main.jl"))

@testset "Square Completion Discovery" begin
    @testset "Data Sampling" begin
        X = sample_states(50, 2)
        @test size(X) == (2, 50)
    end

    @testset "Bridge Loss" begin
        problem = default_square_completion_problem()
        X = problem.state_sampler(80)
        loss = bridge_loss([1.0, 2.0], problem, X)
        @test loss < 1e-12
    end

    @testset "Discovery" begin
        problem = default_square_completion_problem()
        shift, expressions, loss = discover_square_completion(problem; io=IOBuffer(), sample_count=60, grid_values=-2.0:0.25:2.0)
        @test shift ≈ [1.0, 2.0]
        @test loss < 1e-12
        @test expressions == ["u = x + 1", "v = y + 2"]
    end

    @testset "Formatted Result" begin
        problem = default_square_completion_problem()
        shift = [1.0, 2.0]
        expressions = build_translation_expressions(problem, shift)
        @test expressions == ["u = x + 1", "v = y + 2"]
    end
end