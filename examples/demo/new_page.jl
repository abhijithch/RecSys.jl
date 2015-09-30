using Reactive
using JSON
include("ALS.jl")

movie_meta = JSON.parse(readall("movie_info.json"))

getfield(m, x, def="") = get(movie_meta, m, Dict()) |>
    (d -> get(d, x, def))


function showmovie(m)
    poster = image(getfield(m,"Poster", "http://uwatch.to/posters/placeholder.png"), alt=m) |>
       size(120px, 180px)
    desc = vbox( fontsize(1.5em, m),
               getfield(m, "Genre")
               )
    #rating_radio = radiogroup([radio(:rate, "Rate $i") for i in 1:5] , 
    #                          name="Rate this movie"
    #                         )

    #rate = string("Rate me!") |> radio(:rating; toggles=true, disabled=false) 
    rate = slider(0:5; name="Your rating", value=0, editable=true, pin=true, disabled=false, secondaryprogress=0)
                 
    vbox(poster,vskip(1em), desc, vskip(1em), rate)

end


function getmovies(movie_set)
    vbox(intersperse(vbox( vskip(1em), hline(), vskip(1em)), 
                               map(showmovie, movie_set[floor(rand(10)*1000), 2]  )                
                              ) )

end



function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout2")

    movie_numbers =  floor(rand(10)*1000)
    movie_set = readdlm("movies.csv",'\,')
    #movie_tile = map(showmovie, movie_set[movie_numbers, 2])

    vlist0 = vbox(title(1, "To get recommendations, rate some movies first"))

    vlist1 = vbox( intersperse(vbox( vskip(1em), hline(), vskip(1em)), 
                               flex(map(showmovie, movie_set[floor(rand(10)*1000), 2]  ) ) 
                              ) 
                 )
    vlist2 = vbox( intersperse(vbox( vskip(1em), hline(), vskip(1em)), 
                               flex(map(showmovie, movie_set[floor(rand(10)*1000), 2]  )  )              
                              )    
                 )
    vlist3 = vbox( intersperse(vbox( vskip(1em), hline(), vskip(1em)), 
                               flex(map(showmovie, movie_set[floor(rand(10)*1000), 2]  ) )               
                              )    
                 )

    #displayedlist = hbox( vlist0, intersperse(vskip(1em), hline(), vskip(1em)),   map(getmovies, [1:3]))
   
    display = vbox(vlist0, vskip(3em), hbox(flex(vlist1), flex(vlist2), flex(vlist3)) )


end

