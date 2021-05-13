# FiVI

[![Build status](https://travis-ci.com/Omastto1/FiVI.jl.svg?branch=master)](https://travis-ci.com/github/Omastto1/FiVI.jl)
[![Coverage Status](https://coveralls.io/repos/github/Omastto1/FiVI.jl/badge.svg?branch=master)](https://coveralls.io/github/Omastto1/FiVI.jl?branch=master)
[![codecov](https://codecov.io/gh/Omastto1/FiVI.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Omastto1/FiVI.jl)

The package implement finite horizon point-based value iteration algorithm in Julia for solving finite horizon partially observable Markov decision  processes (FH POMDPs). The user should define the problem with [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) and [FiniteHorizonPOMDPs.jl](https://github.com/JuliaPOMDP/FiniteHorizonPOMDPs.jl) interfaces.

## Installation

```julia
using Pkg; Pkg.add("FiVI")
```

## Usage

Given an FH POMDP `pomdp` defined with [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) and [FiniteHorizonPOMDPs.jl](https://github.com/JuliaPOMDP/FiniteHorizonPOMDPs.jl) interfaces, use

```julia
using FiVI
solver = FiVISolver(precision=1., time_limit=50.) # creates the solver
policy = solve(solver, pomdp) # runs solver
```
To extract the policy for a given state, simply call the action function:

```julia
a = action(policy, s, t) # returns the optimal action for state s in a given stage t
```

Or to extract the value, use
```julia
value(policy, s, t) # returns the optimal value of state s in a given stage t
```

## Terminality note
The solver does not check the terminality of states. Thus, it is important to have correctly defined `transition` (terminal states can not move elsewhere than to themselves) as well as `reward` (reward only transition from non-terminal states to terminakl ones).
