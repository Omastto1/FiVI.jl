module FiVI

using POMDPs
using POMDPPolicies
using POMDPModelTools
using LinearAlgebra
using FiniteHorizonPOMDPs
using Distributions
import POMDPs: Solver, solve
import Base: ==, hash, convert, getindex
import FiniteHorizonPOMDPs: InStageDistribution, FixedHorizonPOMDPWrapper

export
    FiVISolver,
    AlphaVec,
    solve

include("solver.jl")


end
