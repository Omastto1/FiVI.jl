struct FiVISolver <: Solver
    precision::Float64
    time_limit::Int64
end

mutable struct Belief
    b::Vector{Float64}
    v::Float64
end

Copy(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

convert(::Type{Array{Float64, 1}}, d::BoolDistribution, pomdp)  = [d.p, 1 - d.p]
convert(::Type{Array{Float64, 1}}, d::DiscreteUniform, pomdp) = [pdf(d, stateindex(pomdp, s)) for s in stage_states(pomdp, 1)]

# Base.getindex(d::BoolDistribution, i::Int64) = i == 1 ? d.p : 1 - d.p
getindex(d::BoolDistribution, i::Int64) = i == 1 ? d.p : 1 - d.p

function Belief(d::BoolDistribution, v::Float64)
    Belief(convert(Array{Float64, 1}, d), v)
end

==(a::Belief, b::Belief) = a.b == b.b
hash(b::Belief) = hash(b.b)

"""
    AlphaVec
Alpha vector type of paired vector and action.
"""
struct AlphaVec
    alpha::Vector{Float64} # alpha vector
    action::Any # action associated wtih alpha vector
end

"""
    AlphaVec(vector, action_index)
Create alpha vector from `vector` and `action_index`.
"""
AlphaVec() = AlphaVec([0.0], 0)

# define alpha vector equality
Base.hash(a::AlphaVec, h::UInt) = hash(a.alpha, hash(a.action, h))
==(a::AlphaVec, b::AlphaVec) = (a.alpha,a.action) == (b.alpha, b.action)


# P(o|b, a) = ∑(sp∈S) P(o|a, sp) ∑(s∈S) P(sp|s, a) * b(s)
function prob_o_given_b_a(pomdp, t, b::Array{Float64}, a, o)
    pr_o_given_b_a = 0.
    for sp in stage_states(pomdp, t + 1)
        temp_sum = sum([pdf(transition(pomdp, s, a), sp) * b[stage_stateindex(pomdp, s)] for s in stage_states(pomdp, t)])
        pr_o_given_b_a += pdf(observation(pomdp, a, sp), o) * temp_sum
    end

    return pr_o_given_b_a
end

function corner_belief(no_states, n)
    b = zeros(no_states)
    b[n] = 1
    return Belief(b, Inf)
end

function b_o_a(pomdp, t, b::Vector{Float64}, a, o, p_o_given_ba)
    b_new = zeros(length(stage_states(pomdp, t + 1)))
    for sp in stage_states(pomdp, t + 1)
        b_temp = 0
        for s in stage_states(pomdp, t)
            b_temp += pdf(transition(pomdp, s, a), sp) * b[stage_stateindex(pomdp, s)]
        end

        b_new[stage_stateindex(pomdp, sp)] = pdf(observation(pomdp, a, sp), o) / p_o_given_ba * b_temp
    end
    println("boa new: $(b_new)")
    return b_new
end

function UB(bp::Array{Float64}, Bs)
    f = []
    c = []
    vs = zeros(length(bp))

    no_corners = []
    # eliminates corner beliefs from iteration and adds others to new list
    for b in Bs
        i = findfirst(b.b .== 1.)
        if i !== nothing
            vs[i] = b.v
        else
            push!(no_corners, b)
        end
    end

    for b in no_corners
        f_temp = b.v
        min_c = Inf
        for s in 1:length(b.b)
            if b.b[s] > 0.
                f_temp -= b.b[s] * vs[s]
                c_temp = bp[s] / b.b[s]
                c_temp < min_c && (min_c = c_temp)
            end
        end
        push!(f, f_temp)
        push!(c, min_c)
    end

    b_opt = argmin(c .* f)
    return c[b_opt] * f[b_opt] + dot(bp, vs)
end

function expand(pomdp, Γs, Bs, BSs, r)
    b = Bs[1][1].b
    for t in 1:horizon(pomdp) - 1
        # println("b: ", b)
        as = []
        for a in ordered_actions(pomdp)
            a_temp = 0.
            for o in stage_observations(pomdp, t)
                prob = prob_o_given_b_a(pomdp, t, b, a, o)
                if prob > 0.
                    a_temp += prob * UB(b_o_a(pomdp, t, b, a, o, prob), Bs[t+1])
                end
            end

            push!(as, a_temp + dot([r(s, a) for s in ordered_stage_states(pomdp, t)], b))
        end

        max_a = ordered_actions(pomdp)[argmax(as)]

        os = []
        os_v = []
        for o in ordered_stage_observations(pomdp, t)
            prob = prob_o_given_b_a(pomdp, t, b, max_a, o)
            if prob > 0.
                boa = b_o_a(pomdp, t, b, max_a, o, prob)
                v = UB(boa, Bs[t+1]) - max([dot(α.alpha, boa) for α in Γs[t+1]]...)
                push!(os, o)
                push!(os_v, v)
            end
        end

        max_o = os[argmax(os_v)]

        boa =  Belief(b_o_a(pomdp, t, b, max_a, max_o, prob_o_given_b_a(pomdp, t, b, max_a, max_o)), Inf)

        if !in(boa, BSs[t+1])
            push!(Bs[t+1], boa)
            push!(BSs[t+1], boa)
        end
        b = boa.b
    end

    return deepcopy(Bs), deepcopy(BSs)
end

function z_k_t_a_o(pomdp, a, o, t, α)
    z = zeros(length(stage_states(pomdp, t)))
    for s in stage_states(pomdp, t)
        z[stage_stateindex(pomdp, s)] = sum([pdf(observation(pomdp, a, sp), o) * pdf(transition(pomdp, s, a), sp) * α.alpha[stage_stateindex(pomdp, sp)] for sp in stage_states(pomdp, t+1)])
    end

    return z
end


function z(pomdp, b, a, t, Γ, r)
    # i think this can be vectorized with r.(s, a)
    r = [r(s, a) for s in stage_states(pomdp, t)] .* b
    if t == horizon(pomdp)
        return r
    else
        temp = zeros(length(stage_states(pomdp, t)))

        for o in stage_observations(pomdp, t)
            zs = [z_k_t_a_o(pomdp, a, o, t, α) for α in Γ]
            temp .+= zs[argmax([dot(b, z) for z in zs])]
        end

        return r .+ temp
    end
end


function backup(pomdp, b, t, Γ, r)
    zs = [z(pomdp, b, a, t, Γ, r) for a in ordered_actions(pomdp)]
    idx = argmax([dot(b, z) for z in zs])
    return AlphaVec(zs[idx], ordered_actions(pomdp)[idx])
end


function upper_bound_update(pomdp, b::Belief, Bs, t, r)
    b.v = -Inf
    for a in ordered_actions(pomdp)
        v_temp = dot([r(s, a) for s in stage_states(pomdp, t)], b.b)

        if t <= horizon(pomdp)
            for o in stage_observations(pomdp, t)
                pr_o_given_b_a = prob_o_given_b_a(pomdp, t, b.b, a, o)
                if pr_o_given_b_a > 0.
                    println(UB(b_o_a(pomdp, t, b.b, a, o, pr_o_given_b_a), Bs))
                    v_temp += pr_o_given_b_a * UB(b_o_a(pomdp, t, b.b, a, o, pr_o_given_b_a), Bs)
                end
            end
        end

        println("values: $(b.v), $(v_temp)")
        b.v = max(b.v, v_temp)
    end

    return b.v
end

function init_belief_space(pomdp, t)
    # add initial belief to first stage
    # add corner beliefs
    no_states = length(stage_states(pomdp, t))
    b_init = convert(Array{Float64, 1}, initialstate(pomdp).d, pomdp)
    if t == 1
        Bs = [Belief(b_init, Inf), [corner_belief(no_states, i) for i in 1:no_states]...]
    else
        Bs = [corner_belief(no_states, i) for i in 1:no_states]
    end
    BSs = Set(Bs)

    # add bonus beliefs
    b = zeros(no_states)
    b = Belief(zeros(no_states), Inf)
    for s in 1:length(stage_states(pomdp, t))
        b.b[s] = b_init[s]
        b_vec = Belief(b.b ./ sum(b.b), Inf)
        if !in(b_vec, BSs)
            push!(Bs, b_vec)
            push!(BSs, b_vec)
        end
    end

    return Bs, BSs
end



function solve(solver::FiVISolver, pomdp::POMDP)
    # init empty Belief and Alpha Vector Lists and belief space set
    Γs, Bs = [AlphaVec[] for i in 1:horizon(pomdp) + 1], [Belief[] for i in 1:horizon(pomdp) + 1]
    BSs = [Set{Belief}() for i in 1:horizon(pomdp) + 1]
    vu = 0.
    time_elapsed = 0

    # initialize belief and alpha vect
    for t in 1:horizon(pomdp) + 1
        Bs[t], BSs[t] = init_belief_space(pomdp, t)
    end

    # ?????????????????????????????????????????????????????????????????????????
    # r = StateActionReward(pomdp)
    r = LazyCachedSAR(pomdp)

    # set upper bound for bleief points in t=T
    for b in Bs[horizon(pomdp) + 1]
        b.v = 0
    end


    while true
        time = @elapsed for t in horizon(pomdp):-1:1
            println("t: ", t, ", horizon: ", horizon(pomdp))
            Γs[t] = []

            for b in Bs[t]
                α = backup(pomdp, b.b, t, Γs[t+1], r)
                push!(Γs[t], α)
            end

            for b in Bs[t]
                b.v = upper_bound_update(pomdp, b, Bs[t + 1], t, r)
            end
        end
        time_elapsed += time
        println("time: ($(time_elapsed))")

        # println("Γs[1]: ", Γs[1])
        # println("Bs[1][1].v: ", Bs[1][1])
        vl = max([dot(α.alpha, Bs[1][1].b) for α in Γs[1]]...)
        vu = Bs[1][1].v
        g_a = 10 ^ (ceil(log10(max(abs(vl), abs(vu)))) - solver.precision)
        # τ' <- time elapsed after start of alg.


        println("\n\n\nvu: ", vu, ", vl: ", vl, "vu - vl = ", vu - vl, ", g_a = ", g_a, "\n\n\n")

        (vu - vl > g_a) & (time_elapsed < solver.time_limit) || break

        Bs, BSs = expand(pomdp, Γs, Bs, BSs, r)
    end

    alphas = Array{Float64, 2}[]
    as = Array{Float64, 1}[]
    for t in 1:size(Γs, 1)
        stage_a_vecs = Γs[t]
        push!(alphas, Array{Float64, 2}(undef, (length(stage_states(pomdp, t)), size(stage_a_vecs, 1))))
        push!(as, Array{Float64, 1}(undef, size(stage_a_vecs, 1)))
        for i in 1:size(stage_a_vecs, 1)
            alphas[t][:, i] = stage_a_vecs[i].alpha
            as[t][i] = stage_a_vecs[i].action
        end
    end

    policy = StagedAlphaVectorPolicy(pomdp, alphas, as)

    return policy, vu
end
