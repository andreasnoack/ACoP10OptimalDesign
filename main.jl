t0 = time()
@info "[Timer: $(round(time() - t0, digits=2))] Loading packages"
using Pumas, LinearAlgebra, Plots, Random, Weave

include("warfarin.jl")
include_weave("hcv.jmd")
