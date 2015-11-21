zero{T<:AbstractString}(::Type{T}) = convert(T, "")

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

type MatFile <: FileSpec
    filename::AbstractString
    entryname::AbstractString
end
read_input(fspec::MatFile) = read(matopen(fspec.filename), fspec.entryname)

type SparseMat <: FileSpec
    S::Union{RatingMatrix, SharedRatingMatrix}
end
read_input(fspec::SparseMat) = fspec.S
