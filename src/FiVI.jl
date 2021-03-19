module FiVI

using POMDPs
using POMDPPolicies
using POMDPModelTools
using LinearAlgebra
using FiniteHorizonPOMDPs
import POMDPs: Solver, solve
import Base: ==, hash, convert, getindex

export
    FiVISolver,
    AlphaVec,
    solve

include("solver.jl")


end
