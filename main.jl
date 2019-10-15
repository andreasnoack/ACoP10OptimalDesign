t0 = time()
@info "[Timer: $(round(time() - t0, digits=2))] Loading packages"
using Pumas, LinearAlgebra

include("warfarin.jl")
