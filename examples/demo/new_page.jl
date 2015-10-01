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
    rating_radio = radiogroup([radio("0", "0"),radio("1", "1"), radio("2", "2"),radio("3", "3"),radio("4", "4"),radio("5", "5")]; 
                              name="Your Rating (0 for \"didn't watch\" )"
                             )

    #rate = slider(0:5; name="Your rating", value=0, editable=true, pin=true, disabled=false, secondaryprogress=0)
                 
    vbox(poster,vskip(1em), desc, vskip(1em), width(20em,rating_radio))

end


function getmovies(movie_set, num_movies)
   # vbox(intersperse(vbox( vskip(1em), hline(), vskip(1em)), 
    #                           map(showmovie, 
        movie_set[floor(rand(num_movies)*1600), 2]                  
   #                           ) ) )

end



function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout2")

    movie_numbers =  floor(rand(10)*1000)
    movie_set = readdlm("movies.csv",'\,')
    #movie_tile = map(showmovie, movie_set[movie_numbers, 2])

    username = textinput("";name=:username, label="Your Name Here", floatinglabel=false, maxlength=256, pattern="(\w)+(\b)(\w)+", error="Please use alphanumerics or _ or space")
    #submit_button = button(      map(pad([left, right], 1em), ["Submit", "Now"]) ; name=:submit, raised=true, disabled=false, noink=true )
    submit_button = iconbutton("send")

    vlist0 = vbox(title(1, "To get recommendations, rate some movies (0 for didn't watch and 1-5 for ratings)"), hskip(3em), width(20em, username))

    vlist1 = hbox( intersperse(hbox( hskip(1em), vline(), hskip(1em)), 
#                               flex(
                   map(showmovie, getmovies(movie_set, 5)) 
# ) 
                              ) 
                 )
    vlist2 = hbox( intersperse(hbox( hskip(1em), vline(), hskip(1em)), 
                               flex(map(showmovie, getmovies(movie_set,5) ) )              
                              )    
                 )
    vlist3 = hbox( intersperse(hbox( hskip(1em), vline(), hskip(1em)), 
                               flex(map(showmovie, getmovies(movie_set,5) ) )               
                              )    
                 )

    #displayedlist = hbox( vlist0, intersperse(vskip(1em), hline(), vskip(1em)),   map(getmovies, [1:3]))
   
    display = vbox(vlist0, vskip(3em),width(10em, submit_button), vskip(3em), vbox(flex(vlist1), flex(vlist2), flex(vlist3)) )

end

