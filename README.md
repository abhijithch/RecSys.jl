How to run::
------------
Example ::

$ julia -p 4_

julia> @everywhere include("src/EveryWhere.jl")

julia> using JuliaRecSys

julia> a = loadData("ml-100k/u1.base",'\t')

julia> x = ALSFactorization(a, 10, 1)
