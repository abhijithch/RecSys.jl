Using RecSys.jl with Netflix dataset.

- dataset required in `.mat` format
- start `nworkers` Julia in parallel `julia -p <nworkers>`
- `include("netflix.jl")`
- run `test("path/to/dataset/netflix.mat", "name_in_mat_file")`
