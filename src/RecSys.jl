module RecSys

using ParallelSparseMatMul
using MAT

if isless(Base.VERSION, v"0.5.0-")
    using SparseVectors
end

import Base: zero

export FileSpec, DlmFile, MatFile, SparseMat, read_input
export ALSWR, train, recommend, rmse, zero
export ParShmem
export save, load

typealias RatingMatrix SparseMatrixCSC{Float64,Int64}
typealias SharedRatingMatrix ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}
typealias InputRatings Union{RatingMatrix,SharedRatingMatrix}
typealias InputIdMap Union{Vector{Int64}, SharedVector{Int64}}
typealias ModelFactor Union{Matrix{Float64}, SharedArray{Float64,2}}
abstract FileSpec

abstract Parallelism
type ParShmem <: Parallelism end

if (Base.VERSION >= v"0.5.0-")
using Base.Threads
type ParThread <: Parallelism end
export ParThread
end

include("input.jl")
include("als_model.jl")
include("als-wr.jl")
include("utils.jl")

# enable logging only during debugging
using Logging
#const logger = Logging.configure(filename="recsys.log", level=DEBUG)
const logger = Logging.configure(level=DEBUG)
logmsg(s) = debug(s)
#logmsg(s) = nothing

end
