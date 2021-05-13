using Test
using LinearAlgebra: dot
using Random
using Distributions
using POMDPs
using FiniteHorizonPOMDPs
using POMDPModelTools
using POMDPModels
using POMDPPolicies
using FiVI
using PointBasedValueIteration

import Base.convert
import FiVI: backup, prob_o_given_b_a, b_o_a, init_belief_space, upper_bound_update, UB, corner_belief, z_k_t_a_o
import FiniteHorizonPOMDPs: InStageDistribution, FixedHorizonPOMDPWrapper

function convert_pbvi(::Type{Array{Float64, 1}}, b::FiniteHorizonPOMDPs.InStageDistribution{Array{Float64, 1}}, pomdp)
    belief = zeros(length(states(pomdp)))
    no_states = length(stage_states(pomdp, 1))
    belief[1 + (b.stage - 1) * no_states : b.stage * no_states] .= b.d
    return belief
end

#test elementary part of backup
@testset "FiVI" begin
    hor = 5
    pomdp = fixhorizon(MiniHallway(), hor)

    @test corner_belief(3, 1).b == [1, 0, 0]

    a = 1
    obs_wrong = (1, 1)
    obs_correct = (1, 4)
    st = 4
    α = AlphaVec([1, 2, 3, 10, 20, 30, 1, 2, 3, 10, 20, 30, 100], 1)

    # observation from wrong stage test
    @test z_k_t_a_o(pomdp, a, obs_wrong, st, α) == zeros(length(stage_states(pomdp, st)))

    # correct observation stage test
    @test z_k_t_a_o(pomdp, a, obs_correct, st, α) == [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.]
end


# test whole backup of tigerPOMDP
@testset "FiVI.jl" begin
    hor = 5
    pomdp = fixhorizon(TigerPOMDP(), hor)
    b_init = convert(Array{Float64, 1}, initialstate(pomdp), pomdp)
    b1 = [1., 0.]; b2 = [0., 1.]
    r = LazyCachedSAR(pomdp)

    # test backups of terminal states
    @test backup(pomdp, b_init, 5, [], r) == AlphaVec([-0.5, -0.5], 0)
    @test backup(pomdp, b1, 5, [], r) == AlphaVec([10., -0.], 1)
    @test backup(pomdp, b2, 5, [], r) == AlphaVec([0., 10.], 2)

    Γ = [AlphaVec([-0.5, -0.5], 0), AlphaVec([10., -0.], 1), AlphaVec([0., 10.], 2)]

    # test backup of stage t <= horizon(t)
    @test backup(pomdp, b_init, 4, Γ, r) == AlphaVec([8., 8.], 0)

    o = (true, 5)
    p = prob_o_given_b_a(pomdp, 5, b_init, 0, o)

    @test p == 0.5

    a = 0
    new_b = b_o_a(pomdp, 5, b_init, a, o, p)

    @test isapprox(new_b, [0.15, 0.85])

    # create states with already backed up values (simpler version)
    Bs, _ = init_belief_space(pomdp, horizon(pomdp)+1)
    for b in Bs
        b.v = -Inf
        for a in ordered_actions(pomdp)
            v_temp = dot([r(s, a) for s in stage_states(pomdp, horizon(pomdp))], b.b)
            b.v = max(b.v, v_temp)
        end
    end

    ub = UB(new_b, Bs)
    @test isapprox(ub, 6.7)

    Bs2, _ = init_belief_space(pomdp, horizon(pomdp))

    v = upper_bound_update(pomdp, Bs2[3], Bs, horizon(pomdp), r)
    @test isapprox(v, 5.7)
end



@testset "Comparison with PBVI" begin
    pomdps = [fixhorizon(MiniHallway(), 5)]

    for pomdp in pomdps
        pomdp = fixhorizon(MiniHallway(), 5)
        solver = FiVISolver(1, 50.)
        policy, vu = solve(solver, pomdp)

        pbvi_solver = PBVISolver(15, 1., false)
        pbvi_policy = solve(pbvi_solver, pomdp)

        @testset "$(typeof(pomdp)) Value function comparison" begin
            B = []
            if typeof(pomdp.m) == MiniHallway
                reachable_beliefs = [([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], 2), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3),
                    ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], 2), ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
                    ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 5),
                    ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], 4),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 4),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 5), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
                    ([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], 3), ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 4), ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
                    ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 4),
                    ([0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 0.0], 1),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], 3), ([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], 4),
                    ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 2),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], 3), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], 2),
                    ([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], 5), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], 4),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], 5),
                    ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5), ([0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.05555555555555555, 0.33333333333333337], 6),
                    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], 3), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4),
                    ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5), ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3),
                    ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4), ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 5),
                    ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5)]
            else
                reachable_beliefs = [(rand(length(stage_states(pomdp, 1))), rand(1:horizon(pomdp))) for _ in 1:100]
            end

            reachable_beliefs_dist = [FiniteHorizonPOMDPs.InStageDistribution(d[1], d[2]) for d in reachable_beliefs]


            fivi_vs = [value(policy, b.d, b.stage) for b in reachable_beliefs_dist]
            pbvi_vs = [value(pbvi_policy,
                        convert_pbvi(Array{Float64, 1},
                        b, pomdp)) for b in reachable_beliefs_dist]

            @test isapprox(fivi_vs, pbvi_vs, rtol=.1)
        end

        @testset "$(typeof(pomdp)) Simulation results comparison" begin
            no_simulations = typeof(pomdp.m) == MiniHallway ? 1 : 10_000
            for s in states(pomdp)
                # Commented parts do not work yet as the POMDPSimulators does not easily support FH
                # println(s)
                # @show value(policy, Deterministic(s))
                # @show value(sarsop_policy, Deterministic(s))
                #
                # @show action(policy, Deterministic(s))
                # @show action(sarsop_policy, Deterministic(s))
                #
                # @show mean([simulate(RolloutSimulator(max_steps = 100), pomdp, policy, updater(policy), Deterministic(s)) for i in 1:no_simulations])
                # @show mean([simulate(RolloutSimulator(max_steps = 100), pomdp, sarsop_policy, updater(sarsop_policy), Deterministic(s)) for i in 1:no_simulations])

                @test isapprox(value(policy, Deterministic(s[1]), s[2]), value(pbvi_policy, Deterministic(s)), rtol=0.1) || value(policy, Deterministic(s[1]), s[2]) >  value(pbvi_policy, Deterministic(s))
                # @test isapprox( mean([simulate(RolloutSimulator(max_steps = 100), pomdp, policy, updater(policy), Deterministic(s)) for i in 1:no_simulations]),
                #                 mean([simulate(RolloutSimulator(max_steps = 100), pomdp, pbvi_policy, updater(pbvi_policy), Deterministic(s)) for i in 1:no_simulations]),
                #                 rtol=0.1)
            end
        end
    end
end
