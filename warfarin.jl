@info "[Timer: $(round(time() - t0, digits=2))] Loading warfarin data"
data = read_pumas(joinpath(@__DIR__(), "data.csv"))

@info "[Timer: $(round(time() - t0, digits=2))] Defining warfarin model"
model = @model begin

  @param begin
    θ₁ ∈ RealDomain(lower=0.0, init=0.15)
    θ₂ ∈ RealDomain(lower=0.0, init=8.0)
    θ₃ ∈ RealDomain(lower=0.0, init=1.0)
    Ω  ∈ PDiagDomain(3)
    σ  ∈ RealDomain(lower=0.0001, init=0.01)
  end

  @random begin
    η ~ MvNormal(Ω)
  end

  @pre begin
    Tvcl = θ₁
    Tvv  = θ₂
    Tvka = θ₃
    CL   = Tvcl*exp(η[1])
    V    = Tvv*exp(η[2])
    Ka   = Tvka*exp(η[3])
  end

  @dynamics OneCompartmentModel

  @vars begin
    conc = Central/V
  end

  @derived begin
    dv ~ @. Normal(log(conc), sqrt(σ))
  end

end

param = (θ₁=0.15,
         θ₂=8.0,
         θ₃=1.0,
         # Ω=[0.07, 0.02, 0.6],
         # Ω=Matrix(Diagonal([0.07, 0.02, 0.6])),
         Ω=Diagonal([0.07, 0.02, 0.6]),
         σ=0.01)


function __expected_information(m::PumasModel,
                                subject::Subject,
                                param::NamedTuple,
                                vrandeffsorth::AbstractVector,
                                ::Pumas.FO,
                                args...; kwargs...)

  trf = Pumas.toidentitytransform(m.param)
  vparam = Pumas.TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  __E_and_V = _param -> Pumas._E_and_V(m, subject, Pumas.TransformVariables.transform(trf, _param), vrandeffsorth, Pumas.FO(), args...; kwargs...)

  # Construct vector of dual numbers for the population parameters to track the partial derivatives
  cfg = Pumas.ForwardDiff.JacobianConfig(__E_and_V, vparam)
  Pumas.ForwardDiff.seed!(cfg.duals, vparam, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  E_d, V_d = __E_and_V(cfg.duals)

  V⁻¹ = inv(Pumas.cholesky(Pumas.ForwardDiff.value.(V_d)))
  dEdθ = hcat((collect(Pumas.ForwardDiff.partials(E_k).values) for E_k in E_d)...)

  m = size(dEdθ, 1)
  n = size(dEdθ, 2)
  dVpart = similar(dEdθ, m, m)
  for l in 1:m
    dVdθl = [Pumas.ForwardDiff.partials(V_d[i,j]).values[l] for i in 1:n, j in 1:n]
    for k in 1:m
      dVdθk = [Pumas.ForwardDiff.partials(V_d[i,j]).values[k] for i in 1:n, j in 1:n]
      # dVpart[l,k] = tr(dVdθk * V⁻¹ * dVdθl * V⁻¹)/2
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  return dEdθ*V⁻¹*dEdθ', dVpart
end

function main()
  tmp1, tmp2 = __expected_information(model, data[1], param, zeros(3), Pumas.FO())
  tmp1[4:7,4:7] = view(tmp2, 4:7,4:7)
  # return det(length(data)*tmp2)
  return rmul!(tmp1, length(data))
end

@info("[Timer: $(round(time() - t0, digits=2))] Computing block diagonal FIM of warfarin model\nDeterminant of FIM: $(det(main()))")
@info("[Timer: $(round(time() - t0, digits=2))] Timing 1000 evaluations of FIM")
@time for i in 1:1000; main(); end
@info("Script completed in $(round(time() - t0, digits=2)) seconds")

