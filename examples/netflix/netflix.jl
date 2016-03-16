using RecSys

function test(matfile, entryname)
    ratings = MatFile(matfile, entryname)
    rec = ALSWR(ratings)
    train(rec, 10, 4)
    err = rmse(rec)
    println("rmse of the model: $err")

    recommended, watched, nexcl = recommend(rec, 100)
    isempty(watched) || println("Already watched: $watched")
    (nexcl == 0) || println("Excluded $(nexcl) movies already watched")
    println("Recommended: $recommended")
    nothing
end
