module RecSys

using ParallelSparseMatMul

import Base: zero

export FileSpec, DlmFile, read_input
export ALSWR, train, recommend, rmse, zero
export save, load

typealias RatingMatrix SparseMatrixCSC{Float64,Int64}
typealias SharedRatingMatrix ParallelSparseMatMul.SharedSparseMatrixCSC{Float64,Int64}
abstract FileSpec

include("als-wr.jl")
include("utils.jl")

# enable logging only during debugging
using Logging
#const logger = Logging.configure(filename="recsys.log", level=DEBUG)
const logger = Logging.configure(level=DEBUG)
logmsg(s) = debug(s)
#logmsg(s) = nothing

end
