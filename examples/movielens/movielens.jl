using RecSys

import RecSys: train, recommend, rmse

type MovieRec
    movie_names::FileSpec
    rec::ALSWR
    movie_mat::Nullable{SparseVector{AbstractString,Int64}}

    function MovieRec(trainingset::FileSpec, movie_names::FileSpec)
        new(movie_names, ALSWR(trainingset), nothing)
    end
end

function movie_names(rec::MovieRec)
    if isnull(rec.movie_mat)
        A = read_input(rec.movie_names)
        movie_ids = convert(Array{Int}, A[:,1])
        movie_names = convert(Array{AbstractString}, A[:,2])
        movie_genres = convert(Array{AbstractString}, A[:,3])
        movies = AbstractString[n*" - "*g for (n,g) in zip(movie_names, movie_genres)]
        M = SparseVector(maximum(movie_ids), movie_ids, movies)
        rec.movie_mat = Nullable(M)
    end

    get(rec.movie_mat)
end

train(movierec::MovieRec, args...) = train(movierec.rec, args...)
rmse(movierec::MovieRec) = rmse(movierec.rec)
recommend(movierec::MovieRec, args...; kwargs...) = recommend(movierec.rec, args...; kwargs...)

function print_recommendations(rec::MovieRec, recommended::Vector{Int}, watched::Vector{Int}, nexcl::Int)
    mnames = movie_names(rec)

    isempty(watched) || println("Already watched:\n$(mnames[watched])")
    (nexcl == 0) || println("Excluded $(nexcl) movies already watched")
    println("Recommended:\n$(mnames[recommended])")
    nothing
end

function test(dataset_path)
    ratings = DlmFile(joinpath(dataset_path, "ratings.csv"), ',', true)
    movies = DlmFile(joinpath(dataset_path, "movies.csv"), ',', true)
    rec = MovieRec(ratings, movies)
    train(rec, 10, 4)
    err = rmse(rec)
    println("rmse of the model: $err")
    print_recommendations(rec, recommend(rec, 100)...)
end
