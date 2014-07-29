# JuliaRecSys

[![Build Status](https://travis-ci.org/abhi123link/JuliaRecSys.jl.png)](https://travis-ci.org/abhi123link/JuliaRecSys.jl)

JuliaRecSys.jl is an implementation of the algorithm in 
Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. Large-Scale Parallel Collaborative Filtering for the Netflix Prize. Proceedings of the 4th international conference on Algorithmic Aspects in Information and Management. Shanghai, China pp. 337-348, 2008
http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf

# Installation

Pkg.clone("https://github.com/thiruk/JuliaRecSys.jl.git")


# Documentation

$ julia -p 4

julia> @everywhere include(".julia/JuliaRecSys/src/EveryWhere.jl")

julia> using JuliaRecSys

julia> a = loadData("input.txt",'\t')

julia> x = ALSFactorization(a, 10, 1)


# Reporting Bugs

# How to contribute

