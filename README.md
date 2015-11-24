# RecSys

[![Build Status](https://travis-ci.org/abhijithch/RecSys.jl.png)](https://travis-ci.org/abhijithch/RecSys.jl)

RecSys.jl is an implementation of the ALS-WR algorithm from
["Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. Large-Scale Parallel Collaborative Filtering for the Netflix Prize. Proceedings of the 4th international conference on Algorithmic Aspects in Information and Management. Shanghai, China pp. 337-348, 2008"](http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf)

## Usage
- Install: `Pkg.clone("https://github.com/abhijithch/RecSys.jl.git")`
- Specify the training dataset in one of several ways:
    - Use delimited (CSV) file with columns: `user_id`, `item_id`, `ratings`. E.g.: `trainingset = DlmFile("ratings.csv", ',', true)`.
    - Use a MAT file, specifying the file and entry name. E.g.: `trainingset = MatFile("ratings.mat", "training")`
    - Provide an implementation of `FileSpec` for any other format.
- Initialize: `als = ALSWR(trainingset)`
- Train: `train(als, num_iterations, num_factors, lambda)`
- Check model quality:
    - `rmse(als)` to check against training dataset
    - `rmse(als, testdataset)` to check against a test dataset
    - and repeat training with different parameters till satisfactory
- Save model: `save(als, filename)`
- Load model: `als = load(filename)`
- Get recommendations:
    - `recommend(als, user_id)` for an existing user
    - `recommend(als, user_ratings)` for a new/anonymous user

## Examples
See examples for more details:
- [last.fm](examples/lastfm/README.md)
- [netflix](examples/netflix/README.md)
- [movielens](examples/movielens/README.md)
