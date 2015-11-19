# RecSys

[![Build Status](https://travis-ci.org/abhijithch/RecSys.jl.png)](https://travis-ci.org/abhijithch/RecSys.jl)

RecSys.jl is an implementation of the ALS-WR algorithm from
["Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan. Large-Scale Parallel Collaborative Filtering for the Netflix Prize. Proceedings of the 4th international conference on Algorithmic Aspects in Information and Management. Shanghai, China pp. 337-348, 2008"](http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf)

# Usage
- Install: `Pkg.clone("https://github.com/abhijithch/RecSys.jl.git")`
- Specify the training dataset.
    - Use delimited (CSV) file with columns: `user_id`, `item_id`, `ratings`. E.g.: `trainingset = DlmFile("ratings.csv", ',', true)`.
    - Provide an implementation of `FileSpec` for any other format.
- Initialize: `als = ALSWR(trainingset)`
- Train: `train(als, num_iterations, num_factors, lambda)`
- Check model error: `rmse(als)`, and repeat training with different parameters till satisfactory
- Save model: `save(als, filename)`
- Load model: `als = load(filename)`
- Get recommendations: `recommend(als, user_id)`

See examples for more details.
