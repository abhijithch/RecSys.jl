include("ALS.jl")
using JSON

movie_meta = JSON.parse(readall("movie_info.json"))

getfield(m, x, def="") = get(movie_meta, m, Dict()) |>
    (d -> get(d, x, def))

function showmovie(m)
    poster = image(getfield(m, "Poster", "http://uwatch.to/posters/placeholder.png"), alt=m) |>
        size(120px, 180px)
    desc = vbox(
        fontsize(1.5em, m),
        caption(getfield(m, "Plot")),
        getfield(m, "Genre")
        )
    hbox(poster, hskip(2em), desc)
end

function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout2")

    userᵗ = Input(1)
    nᵗ = Input(10)
   
    users = ["Alice", "Bob", "Charlie", "Dorothy"]
    lift(userᵗ, nᵗ) do user, n
        vbox(
            dropdownmenu("Who", users, selected=1) >>> userᵗ,
            hbox("Count", slider(1:100, value=10) >>> nᵗ) |>
                packacross(center),
            title(2, "Top $n recommendations for $(users[user])"),
            vskip(2em),
            intersperse(vbox(vskip(1em), hline(), vskip(1em)),
                map(showmovie, recommend(user, n))),
        ) |> pad(1em)
    end
end
