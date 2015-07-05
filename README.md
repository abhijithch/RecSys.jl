# JuliaRecSys

[![Build Status](https://travis-ci.org/abhijithch/RecSys.jl.png)](https://travis-ci.org/abhijithch/RecSys.jl)

RecSys.jl is an implementation of the algorithm from 
["Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. Large-Scale Parallel Collaborative Filtering for the Netflix Prize. Proceedings of the 4th international conference on Algorithmic Aspects in Information and Management. Shanghai, China pp. 337-348, 2008"](http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf)

# Installation

```
Pkg.clone("https://github.com/abhijithch/RecSys.jl.git")
```

# Documentation

```
$ julia -p 4

julia> @everywhere include(".julia/RecSys/src/EveryWhere.jl")

julia> using RecSys

julia> a = loadData("input.txt",'\t')

julia> x = ALSFactorization(a, 10, 1)
```

# Reporting Bugs

https://github.com/abhijithch/RecSys.jl/issues

# How to contribute

