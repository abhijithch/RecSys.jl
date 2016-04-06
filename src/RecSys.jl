module RecSys

using ParallelSparseMatMul
using MAT
using Blobs

include("chunks/matrix.jl")
using .MatrixBlobs

if isless(Base.VERSION, v"0.5.0-")
    using SparseVectors
else
    nonzeroinds = Base.SparseArrays.nonzeroinds
    nonzeros = Base.SparseArrays.nonzeros
end

import Base: zero
import Blobs: save, load

export FileSpec, DlmFile, MatFile, SparseMat, SparseBlobs, DenseBlobs, read_input
export ALSWR, train, recommend, rmse, zero
export ParShmem, ParBlob
export save, load, clear, localize!

typealias RatingMatrix SparseMatrixCSC{Float64,Int64}
typealias SharedRatingMatrix ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}
typealias InputRatings Union{RatingMatrix,SharedRatingMatrix,SparseMatBlobs}
typealias InputIdMap Union{Vector{Int64}, SharedVector{Int64}}
typealias ModelFactor Union{Matrix{Float64}, SharedArray{Float64,2}, DenseMatBlobs{Float64}}

abstract FileSpec
abstract Inputs
abstract Model

abstract Parallelism
type ParShmem <: Parallelism end
type ParBlob <: Parallelism end

if (Base.VERSION >= v"0.5.0-")
using Base.Threads
type ParThread <: Parallelism end
export ParThread
else
threadid() = 1
macro threads(x)
end
end

function tstr()
    t = time()
    string(Libc.strftime("%Y-%m-%dT%H:%M:%S",t), Libc.strftime("%z",t)[1:end-2], ":", Libc.strftime("%z",t)[end-1:end])
end

# enable logging only during debugging
#using Logging
##const logger = Logging.configure(filename="recsys.log", level=DEBUG)
#const logger = Logging.configure(level=DEBUG)
#macro logmsg(s)
#    quote
#        debug("[", myid(), "-", threadid(), "] ", $(esc(s)))
#    end
#end
macro logmsg(s)
end
#macro logmsg(s)
#    quote
#        info(tstr(), " [", myid(), "-", threadid(), "] ", $(esc(s)))
#    end
#end

include("inputs/inputs.jl")
include("models/models.jl")

include("als-wr.jl")
include("utils.jl")

end
