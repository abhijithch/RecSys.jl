<<<<<<< HEAD
# JuliaRecSys

[![Build Status](https://travis-ci.org/abhi123link/JuliaRecSys.jl.png)](https://travis-ci.org/abhi123link/JuliaRecSys.jl)
=======
How to run::
------------
Example ::

$ julia -p 4_

julia> @everywhere include("src/EveryWhere.jl")

julia> using JuliaRecSys

julia> a = loadData("ml-100k/u1.base",'\t')

julia> x = ALSFactorization(a, 10, 1)
>>>>>>> d4d2865b7e8e307d82c45e8f354852e4f7970faa
