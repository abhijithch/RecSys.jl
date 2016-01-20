using RecSys

import RecSys: train, recommend, rmse

if isless(Base.VERSION, v"0.5.0-")
    using SparseVectors
end

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
rmse(movierec::MovieRec, args...; kwargs...) = rmse(movierec.rec, args...; kwargs...)
recommend(movierec::MovieRec, args...; kwargs...) = recommend(movierec.rec, args...; kwargs...)

function print_list(mat::SparseVector, idxs::Vector{Int}, header::AbstractString)
    if isless(Base.VERSION, v"0.5.0-")
        if !isempty(idxs)
            println(header)
            for idx in idxs
                println("[$idx] $(mat[idx])")
            end
        end
    else
        isempty(idxs) || println("$header\n$(mat[idxs])")
    end
end

function print_recommendations(rec::MovieRec, recommended::Vector{Int}, watched::Vector{Int}, nexcl::Int)
    mnames = movie_names(rec)

    print_list(mnames, watched, "Already watched:")
    (nexcl == 0) || println("Excluded $(nexcl) movies already watched")
    print_list(mnames, recommended, "Recommended:")
    nothing
end

function test(dataset_path)
    ratings_file = DlmFile(joinpath(dataset_path, "ratings.csv"); dlm=',', header=true)
    movies_file = DlmFile(joinpath(dataset_path, "movies.csv"); dlm=',', header=true)
    rec = MovieRec(ratings_file, movies_file)
    train(rec, 10, 4)

    err = rmse(rec)
    println("rmse of the model: $err")

    println("recommending existing user:")
    print_recommendations(rec, recommend(rec, 100)...)

    println("recommending anonymous user:")
    u_idmap = RecSys.user_idmap(rec.rec.inp)
    i_idmap = RecSys.item_idmap(rec.rec.inp)
    # take user 100
    actual_user = findfirst(u_idmap, 100)
    rated_anon, ratings_anon = RecSys.items_and_ratings(rec.rec.inp, actual_user)
    actual_movie_ids = i_idmap[rated_anon]
    sp_ratings_anon = SparseVector(maximum(i_idmap), actual_movie_ids, ratings_anon)
    print_recommendations(rec, recommend(rec, sp_ratings_anon)...)

    println("saving model to model.sav")
    save(rec, "model.sav")
    nothing
end
