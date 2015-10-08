
using JSON

include("./ALSDemo.jl")

getfield(meta, m, x, def="") = get(meta, m, Dict()) |>
    (d -> get(d, x, def))

function showmovie(meta, m)
    poster = image(getfield(meta, m, "Poster", "http://uwatch.to/posters/placeholder.png"), alt=m) |>
        size(120px, 180px)
    desc = vbox(
        fontsize(1.5em, m),
        caption(getfield(meta, m, "Plot")),
        getfield(meta, m, "Genre")
        )
    hbox(poster, hskip(2em), desc)
end

function show_list(user, R, U, M, n, meta)
    intersperse(vbox(vskip(1em), hline(), vskip(1em)),
        map(m -> showmovie(meta, m), recommend(U,M,R,user, n))) |> vbox
end

function main(window)
    push!(window.assets, "widgets")
    push!(window.assets, "layout2")

    userᵗ = Input(1)
    nᵗ = Input(10)
    R = Rating("./data/ml-100k/u1.base",'\t',false)
    U , M = factorize(R,10,10)
    users = ["Alice", "Bob", "Charlie", "Dorothy"]
    movie_meta = JSON.parse(readall("./data/movie_info.json"))

    lift(userᵗ, nᵗ) do user, n
        mnu = vbox(
            dropdownmenu("Who", menu(users) >>> userᵗ),
            hbox("Count", slider(1:100, value=10) >>> nᵗ) |>
                packacross(center),
            title(2, "Top $n recommendations for $(users[user])"),
            vskip(2em),
        ) |> pad(1em)

        vbox(mnu, show_list(user, R, U, M, n, movie_meta))
    end
end
