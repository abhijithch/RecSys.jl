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

# sparse matrix blobs (input only)
type SparseBlobs <: FileSpec
    name::AbstractString
    maxcache::Int
    function SparseBlobs(name::AbstractString; maxcache::Int=10)
        new(name, maxcache)
    end
end
read_input(fspec::SparseBlobs) = SparseMatBlobs(fspec.name; maxcache=fspec.maxcache)

# dense matrix blobs (input and output)
type DenseBlobs <: FileSpec
    name::AbstractString
    maxcache::Int
    function DenseBlobs(name::AbstractString; maxcache::Int=10)
        new(name, maxcache)
    end
end
read_input(fspec::DenseBlobs) = DenseMatBlobs(fspec.name; maxcache=fspec.maxcache)

const MAX_BLK_BYTES = 128*1000*1000 #128MB
function _max_items(T::Type, D::Int, sz::Tuple)
    m,n = sz
    unsplit_dim = (D == 1) ? n : m
    ceil(Int, MAX_BLK_BYTES/sizeof(T)/unsplit_dim)
end
function create{T}(fspec::DenseBlobs, ::Type{T}, D::Int, sz::Tuple, init::Function, max_items::Int=_max_items(T,D,sz))
    @logmsg("creating densematarray")
    isdir(fspec.name) || mkdir(fspec.dir)
    m,n = sz
    unsplit_dim = (D == 1) ? n : m
    split_dim = sz[D]
    dm = DenseMatBlobs(T, D, unsplit_dim, fspec.name)

    startidx = 1
    while startidx <= split_dim
        idxrange = startidx:min(split_dim, startidx + max_items)
        blobsz = (D == 1) ? (length(idxrange),unsplit_dim) : (unsplit_dim,length(idxrange))
        M = init(T, blobsz...)
        @logmsg("idxrange: $idxrange, sz: $(size(M))")
        append!(dm, M)
        startidx = last(idxrange) + 1
    end
    dm
end
