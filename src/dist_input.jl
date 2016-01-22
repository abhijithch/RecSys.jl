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
    R::Nullable{ChunkedFile}
    RT::Nullable{ChunkedFile}

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
        R = read_input(inp.Rfile)
        RT = read_input(inp.RTfile)

        last_chunk = R.chunks[end]
        inp.nitems = last(last_chunk.keyrange)
        last_chunk = RT.chunks[end]
        inp.nusers = last(last_chunk.keyrange)

        inp.R = R
        inp.RT = RT
    end
    nothing
end

item_idmap(inp::DistInputs) = Int64[]
user_idmap(inp::DistInputs) = Int64[]

nusers(inp::DistInputs) = inp.nusers
nitems(inp::DistInputs) = inp.nitems

users_and_ratings(inp::DistInputs, i::Int64) = _sprowsvals(get(inp.R), i)
all_user_ratings(inp::DistInputs, i::Int64) = _spvals(get(inp.R), i)
all_users_rated(inp::DistInputs, i::Int64) = _sprows(get(inp.R), i)

items_and_ratings(inp::DistInputs, u::Int64) = _sprowsvals(get(inp.RT), u)
all_item_ratings(inp::DistInputs, u::Int64) = _spvals(get(inp.RT), u)
all_items_rated(inp::DistInputs, u::Int64) = _sprows(get(inp.RT), u)

function _sprowsvals{K,SK,SV}(cf::ChunkedFile{K,SparseMatrixCSC{SK,SV}}, col::Int64)
    chunk = getchunk(cf, col)
    d = data(chunk, cf.lrucache)
    _sprowsvals(d, col - first(chunk.keyrange) + 1)
end

function _sprows{K,SK,SV}(cf::ChunkedFile{K,SparseMatrixCSC{SK,SV}}, col::Int64)
    chunk = getchunk(cf, col)
    d = data(chunk, cf.lrucache)
    _sprows(d, col - first(chunk.keyrange) + 1)
end

function _spvals{K,SK,SV}(cf::ChunkedFile{K,SparseMatrixCSC{SK,SV}}, col::Int64)
    chunk = getchunk(cf, col)
    d = data(chunk, cf.lrucache)
    _spvals(d, col - first(chunk.keyrange) + 1)
end
