# cd("../..")
# push!(LOAD_PATH, "./src")

# cd /Users/adelinehillier/.julia/dev/OceanConvect

using OceanConvect
using Plots

# Construct a ProfileData object, 𝒟, consisting of the data to train on,
# and a GP object, 𝒢, with which to perform the regression.

# data
filename = "general_strat_32_profiles.jld2"
problem  = Sequential("T") # v: "T" for temperature profile; "wT" for temperature flux profile
D        = 16       # collapse profile data down to 16 gridpoints
N        = 4        # collect every 4 timesteps' data for training

# Now let's define our kernel. We don't know what the best long(length-scale) parameter value is yet so let's guess 1.0 (γ=10) and see what happens.
# Let's use an exponential kernel.

# kernel
k        = 2        # kernel function ID
logγ     = 1.0      # log(length-scale parameter)
logσ     = 0.0      # log(signal variance parameter)
distance = derivative_distance  # distance metric to use in the kernel
kernel   = get_kernel(k, logγ, logσ, distance)

# data
𝒟 = OceanConvect.ModelData.data(filename, problem; D=D, N=N)

# model
𝒢 = OceanConvect.GaussianProcess.model(𝒟; kernel = kernel)

# Animate the mean GP prediction.
anim = animate_profile(𝒢, 𝒟)
gif(anim, "animated_profile_γ_1.5.gif")

# Not great. Let's try to optimize our γ value.
# We can use get_min_gamma function to search the range -3:0.1:3 for the log(γ) value that minimizes the mean error on the true check
min_log_param, min_error = get_min_gamma(k, 𝒟, distance, -3:0.1:3)

# Let's create a kernel with the better log(γ) value, put the new kernel into a new model, and see how it does.
new_kernel = get_kernel(k, min_log_param, logσ, distance)
𝒢 = OceanConvect.GaussianProcess.model(𝒟; kernel = new_kernel)
anim = animate_profile(𝒢, 𝒟)
gif(anim, "animated_profile_optimized.gif")

# There's clearly a lot of flexibility in how we design our kernel function
