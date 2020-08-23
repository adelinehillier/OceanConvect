cd("../..")
push!(LOAD_PATH, "./src")

##
using OceanConvect

# Construct a ProfileData object, 𝒟, consisting of the data to train on,
# and a GP object, 𝒢, with which to perform the regression.

# data
filename = "general_strat_4_profiles.jld2"
problem  = SequentialProblem("dT") # v: "T" for temperature profile; "wT" for temperature flux profile
D        = 16       # collapse profile data down to 16 gridpoints
N        = 4        # collect every 4 timesteps' data for training

# kernel
k        = 2        # kernel function ID
logγ     = 4.1      # log(length-scale parameter)
logσ     = 0.0      # log(signal variance parameter)
distance = derivative_distance  # distance metric to use in the kernel
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟 = data(filename, problem; D=D, N=N)

# model
𝒢 = GP.model(𝒟; kernel = kernel)

# Find the log(length-scale parameter) value in the range -3:0.1:3 that minimizes the mean error on the true check
min_log_param, min_error = get_min_gamma(k, 𝒟, d, -3:0.1:3)

# Animate the mean GP prediction.
anim = animate_profile(𝒢, 𝒟, v)
gif(anim, "./figures/animated_profile.gif")
