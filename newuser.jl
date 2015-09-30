using Reactive
include("index.jl")

movie_meta = JSON.parse(readall("movie_info.json"))

getfield(m, x, def="") = get(movie_meta, m, Dict()) |>
    (d -> get(d, x, def))


function showmovie(m)
    poster = image(getfield(m, "http://uwatch.to/posters/placeholder.png"), alt=m) |>
       size(120px, 180px)
    desc = vbox( fontsize(1.5em, m),
               getfield(m, "Genre")
               )
    rating_radio = radiogroup(vec(radio([string("rate", i) for i=1:5]) ),
                              name="Rate this movie"
                             )

end



function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout")

    vlist1 = vbox()
    vlist2 = vbox()
    vlist3 = vbox()


    hbox(vlist1, vlist2, vlist3)






end
