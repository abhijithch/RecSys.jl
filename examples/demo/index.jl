include("ALS.jl")

function showmovie(title)
    fontsize(1.5em, title)
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
