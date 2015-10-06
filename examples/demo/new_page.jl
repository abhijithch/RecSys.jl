using Reactive
using JSON
include("ALS.jl")

movie_meta = JSON.parse(readall("./data/movie_info.json"))

getfield(m, x, def="") = get(movie_meta, m, Dict()) |>
    (d -> get(d, x, def))

function showmovie(m)
    poster = image(getfield(m,"Poster", "http://uwatch.to/posters/placeholder.png"), alt=m) |>
       size(120px, 180px)
    desc = vbox( fontsize(1em, m), getfield(m, "Genre")    )

    #rating_widget = radiogroup([radio("0", "0"),radio("1", "1"), radio("2", "2"),radio("3", "3"),radio("4", "4"),radio("5", "5")]; 
    #                          name="Your Rating (0 for \"didn't watch\" )"
    #                         )

    user_rating = Input(0)
    rating_widget = subscribe( slider(0:5; name="Your rating", value=0, editable=true, pin=true, disabled=false, secondaryprogress=0), 
                               user_rating )

    #user_ratingᵗ = lift(user_rating) do 
    #end

    vbox(poster,vskip(0.7em),
         getfield(m, "Title") |> fontsize(1.2em), vskip(0.5em),  
         getfield(m, "Genre") |> fontsize(0.7em), 
         vskip(1em), width(15em,rating_widget))

end 

#=
function updaterating!(user_rating)
end

function ratingwidget(movietile, movielist)

    user_rating = Input(0)
    rating_widget = subscribe(slider(0:5; name="Your rating", value=0, editable=true, pin=true, disabled=false, secondaryprogress=0),
                              user_rating ) 
    user_ratingᵗ = lift(user_rating) do
    end

    vbox(movietile, vskip(1em), rating_widget)
end
=#


function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout2")

    movie_dataset = readdlm("movies.csv",'\,')

    username = textinput("";name=:username, label="Your Name Here", floatinglabel=false, maxlength=256, pattern="(\w)+(\b)(\w)+", error="Please use alphanumerics or _ or space")
    #submit_button = button(      map(pad([left, right], 1em), ["Submit", "Now"]) ; name=:submit, raised=true, disabled=false, noink=true )
    submit_button = iconbutton("send")

    vlist0 = vbox(title(1, "To get recommendations, rate some movies (0 for didn't watch and 1-5 for ratings)"), hskip(3em), width(20em, username))

### GET A LIST OF MOVIES, num_movies IS THE NUMBER OF MOVIES YOU WANT ###
## nrows is the number of rows in the display, and ncols is the number of columns per row.

    nrows = 3
    ncols = 5
    num_movies = nrows*ncols
    movielist = movie_dataset[floor(rand(num_movies)*1600), :]
    #ratingvec = zeros(size(movie_dataset)) 

    vlist1 = hbox( intersperse(hbox( hskip(1em), vline(), hskip(1em) ), 
                                     grow(   # map( ratingwidget, 
                                               map(showmovie, movielist[1:ncols,2])
                                             # )
                                         )                                    
                              ) 
                 )


    vlist2 = hbox( intersperse(hbox( hskip(1em), vline(), hskip(1em)), 
                               grow( map(showmovie, movielist[ncols+1:2*ncols,2] ) )              
                              )    
                 )

    display = vbox(vlist0, vskip(3em),width(10em, submit_button), vskip(3em), vbox(vlist1), vskip(2em), hline(), vskip(2em), vbox(vlist2) )

end

