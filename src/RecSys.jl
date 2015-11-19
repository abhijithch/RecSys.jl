module RecSys

using ParallelSparseMatMul

import Base: zero

export FileSpec, DlmFile, read_input
export ALSWR, train, recommend, rmse, zero
export save, load

include("als-wr.jl")
include("utils.jl")

# enable logging only during debugging
using Logging
#const logger = Logging.configure(filename="recsys.log", level=DEBUG)
const logger = Logging.configure(level=DEBUG)
logmsg(s) = debug(s)
#logmsg(s) = nothing

end
