using Test
import MyPackage: myiseven

@testset "myiseven" begin
    @test myiseven(1) == false
    @test myiseven(2) == true
    @test myiseven(3) == false
end

nothing # test実行時に標準出力に大量のメッセージが出ないようにするおまじない