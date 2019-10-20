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

data_full_profile = simobs(model, data[1:5], param, obstimes=range(0, stop=last(data[1].time), length=1000))
plot(data_full_profile[1].times, [data_full_profile[i].observed.conc for i in 1:5], title="Warfarin", xlabel="time (in hours)", ylabel="concentration (mg/L)", linewidth=3)

fim() = Pumas._expected_information(model, data[1], param, zeros(3), Pumas.FO())

@info("[Timer: $(round(time() - t0, digits=2))] Computing block diagonal FIM of warfarin model\nDeterminant of FIM: $(logdet(fim()[1]*length(data)))")
@info("[Timer: $(round(time() - t0, digits=2))] Timing 1000 evaluations of FIM")
@time for i in 1:1000; fim(); end
@info("Script completed in $(round(time() - t0, digits=2)) seconds")

