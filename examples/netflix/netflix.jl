using RecSys

function test(matfile, entryname)
    ratings = MatFile(matfile, entryname)
    rec = ALSWR(ratings, ParShmem())
    train(rec, 10, 4)
    err = rmse(rec)
    println("rmse of the model: $err")

    recommended, watched, nexcl = recommend(rec, 100)
    isempty(watched) || println("Already watched: $watched")
    (nexcl == 0) || println("Excluded $(nexcl) movies already watched")
    println("Recommended: $recommended")
    nothing
end

function test_chunks(dataset_path, model_path)
    mem = Base.Sys.free_memory()
    mem_model = mem_inputs = round(Int, mem/3)
    user_item_ratings = SparseBlobs(joinpath(dataset_path, "splits", "R_itemwise"); maxcache=mem_model)
    item_user_ratings = SparseBlobs(joinpath(dataset_path, "splits", "RT_userwise"); maxcache=mem_model)

    rec = ALSWR(user_item_ratings, item_user_ratings, ParBlob())

    train(rec, 10, 4, model_path, mem_inputs)

    err = rmse(rec)
    println("rmse of the model: $err")
    nothing
end
