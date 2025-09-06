include("helper.jl")

module NIRBeer

export add_two, scale

add_two(x) = x + 2
add_two_helper(x) = helper(x) + 2
scale(v, a) = a .* v

end