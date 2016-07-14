type SharedMemoryInputs <: Inputs
    ratings_file::FileSpec
    nusers::Int
    nitems::Int
    R::Nullable{InputRatings}
    RT::Nullable{InputRatings}
    item_idmap::Nullable{InputIdMap}
    user_idmap::Nullable{InputIdMap}

    function SharedMemoryInputs(file::FileSpec)
        new(file, 0, 0, nothing, nothing, nothing, nothing)
    end
end

function clear(inp::SharedMemoryInputs)
    inp.R = nothing
    inp.RT = nothing
    inp.item_idmap = nothing
    inp.user_idmap = nothing
end

function localize!(inp::SharedMemoryInputs)
    if !isnull(inp.R)
        R = get(inp.R)
        isa(R, SharedRatingMatrix) && (inp.R = copy(R))
    end

    if !isnull(inp.RT)
        RT = get(inp.RT)
        isa(RT, SharedRatingMatrix) && (inp.RT = copy(RT))
    end

    if !isnull(inp.item_idmap)
        item_idmap = get(inp.item_idmap)
        isa(item_idmap, SharedVector) && (inp.item_idmap = copy(item_idmap))
    end

    if !isnull(inp.user_idmap)
        user_idmap = get(inp.user_idmap)
        isa(user_idmap, SharedVector) && (inp.user_idmap = copy(user_idmap))
    end
    nothing
end

function share!(inp::SharedMemoryInputs)
    if !isnull(inp.R)
        R = get(inp.R)
        isa(R, SharedRatingMatrix) || (inp.R = share(R))
    end

    if !isnull(inp.RT)
        RT = get(inp.RT)
        isa(RT, SharedRatingMatrix) || (inp.RT = share(RT))
    end

    if !isnull(inp.item_idmap)
        item_idmap = get(inp.item_idmap)
        isa(item_idmap, SharedVector) || (inp.item_idmap = share(item_idmap))
    end

    if !isnull(inp.user_idmap)
        user_idmap = get(inp.user_idmap)
        isa(user_idmap, SharedVector) || (inp.user_idmap = share(user_idmap))
    end

    nothing
end

# Note:
# Filtering out causes the item and user ids to change.
# We need to keep a mapping to be able to match in the recommend step.
function filter_empty(R::RatingMatrix; only_items=Int64[])
    if !isempty(only_items)
        max_num_items = maximum(only_items)
        if size(R, 2) < max_num_items
            RI, RJ, RV = findnz(R)
            R = sparse(RI, RJ, RV, size(R, 1), max_num_items)
        end
        R = R'
        R = R[only_items, :]
        R = R'
        non_empty_items = only_items
    end

    U = sum(R, 2)
    non_empty_users = find(U)
    R = R[non_empty_users, :]

    if isempty(only_items)
        R = R'
        P = sum(R, 2)
        non_empty_items = find(P)
        R = R[non_empty_items, :]
        R = R'
    end

    R, non_empty_items, non_empty_users
end

function ensure_loaded(inp::SharedMemoryInputs; only_items=Int64[])
    if isnull(inp.R)
        @logmsg("loading inputs...")
        t1 = time()
        A = read_input(inp.ratings_file)

        if isa(A, SparseMatrixCSC)
            R = convert(SparseMatrixCSC{Float64,Int64}, A)
        else
            # separate the columns and make them of appropriate types
            users   = convert(Vector{Int64},   A[:,1])
            items   = convert(Vector{Int64},   A[:,2])
            ratings = convert(Vector{Float64}, A[:,3])

            # create a sparse matrix
            R = sparse(users, items, ratings)
        end

        R, item_idmap, user_idmap = filter_empty(R; only_items=only_items)
        inp.R = R
        inp.nusers, inp.nitems = size(R)
        inp.item_idmap = (extrema(item_idmap) == (1,length(item_idmap))) ? nothing : item_idmap
        inp.user_idmap = (extrema(user_idmap) == (1,length(user_idmap))) ? nothing : user_idmap
        inp.RT = R'
        t2 = time()
        isnull(inp.item_idmap) && @logmsg("no need to map item_ids")
        isnull(inp.user_idmap) && @logmsg("no need to map user_ids")
        @logmsg("time to load inputs: $(t2-t1) secs")
    end
    nothing
end

item_idmap(inp::SharedMemoryInputs) = isnull(inp.item_idmap) ? Int64[] : get(inp.item_idmap)
user_idmap(inp::SharedMemoryInputs) = isnull(inp.user_idmap) ? Int64[] : get(inp.user_idmap)
