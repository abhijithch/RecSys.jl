# For very large inputs, it is practical to preprocess data as much as possible.
# So in distributed memory mode, we expect inputs to:
#   - have normalized all user and item ids to be integers starting from 1
#   - have user-item and item-user (the transpose) rating data pre-created so that we don't have to transpose a large matrix every time
#   - have filtered out all empty users and items from the data
type DistInputs <: Inputs
    Rfile::FileSpec
    RTfile::FileSpec
    nusers::Int
    nitems::Int
    R::Nullable{InputRatings}
    RT::Nullable{InputRatings}

    function DistInputs(ratings_file::FileSpec, transposed_ratings_file::FileSpec)
        new(ratings_file, transposed_ratings_file, 0, 0, nothing, nothing)
    end
end

function clear(inp::DistInputs)
    inp.R = nothing
    inp.RT = nothing
end

localize!(inp::DistInputs) = nothing
share!(inp::DistInputs) = nothing

function ensure_loaded(inp::DistInputs)
    if isnull(inp.R)
        # R is the user x item matrix (users in rows and items in columns)
        R = read_input(inp.Rfile)
        RT = read_input(inp.RTfile)
        inp.nusers, inp.nitems = size(R)

        inp.R = R
        inp.RT = RT
    end
    nothing
end

item_idmap(inp::DistInputs) = Int64[]
user_idmap(inp::DistInputs) = Int64[]
