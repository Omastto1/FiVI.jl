using Test
using POMDPs
using POMDPModelTools
using POMDPModels
using FiVI
using FiniteHorizonPOMDPs
using LinearAlgebra: dot

import FiVI: backup, prob_o_given_b_a, b_o_a, init_belief_space, upper_bound_update, UB


# @testset "FiVI" begin
    # hor = 5
    # pomdp = fixhorizon(MiniHallway(), hor)
    #
    # @test corner_belief(3, 1).b == [1, 0, 0]
    #
    # a = 1
    # o = (Observation(1), 1)
    # t = 4
    # α = AlphaVec([1, 2, 3, 10, 20, 30, 1, 2, 3, 10, 20, 30, 100], 1)
    #
    # @test z_k_t_a_o(pomdp, a, o, t, α) == zeros(length(stage_states(pomdp, t)))
    #
    #
    # o = (Observation(1), 4)
    #
    # # sum([pdf(observation(pomdp, a, sp), o) * pdf(transition(pomdp, s, a), sp) * α.alpha[stage_stateindex(pomdp, s)] for sp in stage_states(pomdp, t+1)])
    #]
    # # s 1 t -> s 1 => 1 * 1 * 1 = 1.
    # # s 2 t -> s 2 => 0 * 1 * 2 = 0.
    # # s 3 t -> s 7 => 0 * ..... = 0.
    # # s 4 t -> s 4 => 0. * .... = 0.
    # # s 5 t -> s 1 => 1 * 1 * 20 = 20
    # # s 6 t -> s 10 => 0 * ..... ]= 0.
    # # s 7 t -> s 1 => 1 * 1 * 1 = 1.
    # # s 8 t -> s 8 => 0 * ..... = 0.
    # # s 9 t -> s 9 => 0 * ..... = 0.
    # # s 10 t -> s 10 => 0 * ... = 0.
    # # s 11 t -> s 13 => 0 * ... = 0.
    # # s 12 t -> s 8 => 0 * .... = 0.
    # # s 13 t -> s all
    # #           s 1 => 1 * 0.08333 * 100 = 8.3337
    #
    # @test z_k_t_a_o(pomdp, a, o, t, α) == [1., 0., 0., 0., 20., 0., 1., 0., 0., 0., 0., 0., 8.3337]
    #
    # no_states = length(stage_states(pomdp, 1))
    # b = corner_belief(no_states, 1).b
    # r = LazyCachedSAR(pomdp)
    # println(backup(pomdp, b, horizon(pomdp), [], r))
# end

@testset "FiVI.jl" begin
    hor = 5
    pomdp = fixhorizon(TigerPOMDP(), hor)
    b_init = convert(Array{Float64, 1}, initialstate(pomdp).d)
    b1 = [1., 0.]; b2 = [0., 1.]
    r = LazyCachedSAR(pomdp)


    # test backups of terminal states
    @test backup(pomdp, b_init, 5, [], r) == AlphaVec([-0.5, -0.5], 0)
    @test backup(pomdp, b1, 5, [], r) == AlphaVec([10., 0.], 2)
    @test backup(pomdp, b2, 5, [], r) == AlphaVec([0., 10.], 1)

    Γ = [AlphaVec([-0.5, -0.5], 0), AlphaVec([10., 0.], 2), AlphaVec([0., 10.], 1)]

    # test b`ackup of stage t <= horizon(t)
    @test backup(pomdp, b_init, 4, Γ, r) == AlphaVec([8., 8.], 0)

    o = (true, 5)

    p = prob_o_given_b_a(pomdp, 5, b_init, 0, o)
    @test p == 0.5

    a = 0

    new_b = b_o_a(pomdp, 5, b_init, a, o, p)

    @test isapprox(new_b, [0.15, 0.85])

    # create states with already backed up values (simpler version)
    Bs, _, _ = init_belief_space(pomdp, horizon(pomdp)+1)
    for b in Bs
        b.v = -Inf
        for a in ordered_actions(pomdp)
            v_temp = dot([r(s, a) for s in stage_states(pomdp, horizon(pomdp))], b.b)
            b.v = max(b.v, v_temp)
        end
    end

    ub = UB(new_b, Bs)
    @test isapprox(ub, 6.7)

    Bs2, _, _ = init_belief_space(pomdp, horizon(pomdp))

    v = upper_bound_update(pomdp, Bs2[3], Bs, horizon(pomdp), r)
    @test isapprox(v, 5.7)
end
