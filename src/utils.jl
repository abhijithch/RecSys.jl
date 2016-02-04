zero{T<:AbstractString}(::Type{T}) = convert(T, "")

# TODO: optimize save/load instead of a blind serialize call
function save(model, filename::AbstractString)
    open(filename, "w") do f
        serialize(f, model)
    end
    nothing
end

function load(filename::AbstractString)
    open(filename, "r") do f
        deserialize(f)
    end
end

##
# Types of file specifications

# DLM/CSV file (input only)
type DlmFile <: FileSpec
    name::AbstractString
    dlm::Char
    header::Bool
    quotes::Bool

    function DlmFile(name::AbstractString; dlm::Char=Base.DataFmt.invalid_dlm(Char), header::Bool=false, quotes::Bool=true)
        new(name, dlm, header, quotes)
    end
end

function read_input(fspec::DlmFile)
    # read file and skip the header
    F = readdlm(fspec.name, fspec.dlm, header=fspec.header, quotes=fspec.quotes)
    fspec.header ? F[1] : F
end

# MAT file (input only)
type MatFile <: FileSpec
    filename::AbstractString
    entryname::AbstractString
end
read_input(fspec::MatFile) = read(matopen(fspec.filename), fspec.entryname)

# sparse matrix - in memory (input only)
type SparseMat <: FileSpec
    S::Union{RatingMatrix, SharedRatingMatrix}
end
read_input(fspec::SparseMat) = fspec.S

# sparse matrix chunks (input only)
type SparseMatChunks <: FileSpec
    metafile::AbstractString
    max_cache::Int

    function SparseMatChunks(metafile::AbstractString, max_cache::Int=5)
        new(metafile, max_cache)
    end
end
read_input(fspec::SparseMatChunks) = ChunkedFile(fspec.metafile, UnitRange{Int64}, SparseMatrixCSC{Float64,Int}, fspec.max_cache)

# dense matrix chunks (input and output)
type DenseMatChunks <: FileSpec
    metafile::AbstractString
    D::Int
    sz::Tuple
    max_cache::Int

    function DenseMatChunks(metafile::AbstractString, splitdim::Int, sz::Tuple, max_cache::Int=5)
        new(metafile, splitdim, sz, max_cache)
    end
end
function read_input(fspec::DenseMatChunks)
    D = fspec.D
    N = fspec.sz[(D==1) ? 2 : 1]
    ChunkedFile(fspec.metafile, UnitRange{Int64}, MemMappedMatrix{Float64,D,N}, fspec.max_cache, false)
end

zero!(a) = fill!(a, 0)

const MAX_BLK_BYTES = 128*1000*1000 #128MB
function _max_items(fspec::DenseMatChunks)
    D = fspec.D
    N = fspec.sz[(D==1) ? 2 : 1]
    ceil(Int, MAX_BLK_BYTES/sizeof(Float64)/N)
end

function create(fspec::DenseMatChunks, initfn::Function=zero!, max_items::Int=_max_items(fspec))
    touch(fspec.metafile)
    cf = read_input(fspec)

    D = fspec.D
    N = fspec.sz[(D==1) ? 2 : 1]
    V = fspec.sz[D]
    NC = ceil(Int, V/max_items)
    chunkpfx = splitext(fspec.metafile)[1]
    empty!(cf.chunks)
    empty!(cf.lrucache)

    for idx in 1:NC
        chunkfname = "$(chunkpfx).$(idx)"
        isfile(chunkfname) && rm(chunkfname)
        r1 = (idx-1)*max_items + 1
        r2 = min(idx*max_items, V)
        M = r2-r1+1
        sz = M*N*sizeof(Float64)
        @logmsg("creating chunk $chunkfname sz: $sz, r: $r1:$r2, N:$N")
        chunk = Chunk(chunkfname, 0, sz, r1:r2, MemMappedMatrix{Float64,D,N})
        push!(cf.chunks, chunk)
    end
    for idx in 1:NC
        r1 = (idx-1)*max_items + 1
        chunk = getchunk(cf, r1)
        A = data(chunk, cf.lrucache)
        initfn(A.val)
        sync!(chunk)
    end
    writemeta(cf)
    nothing
end
