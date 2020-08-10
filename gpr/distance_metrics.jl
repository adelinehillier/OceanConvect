# Distance metrics for kernel functions

δ(Φ, z) = diff(Φ) ./ diff(z)

# this is norm(a-b)^2 but more efficient
function sq_mag(a,b) # ||a - b||^2
    ll = 0.0
    indices = 1:length(a)
    @inbounds for k in indices
        ll += (a[k]-b[k])^2
    end
    return ll
end

"""
l2_norm: computes the Euclidean distance (l²-norm) between two vectors
"""
function l2_norm(a,b,z) # d(x,x') = || x - x' ||
    return sqrt(sq_mag(a,b))
end

function l2_norm(a,b) # d(x,x') = || x - x' ||
    return sqrt(sq_mag(a,b))
end

"""
h1_norm: computes the H¹-norm w.r.t z of two vectors
"""
function h1_norm(a,b,z) # d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||
    return l2_norm( δ(a, z), δ(b, z) )
end

"""
hm1_norm: computes the H⁻¹-norm w.r.t z of two vectors
"""
function hm1_norm(a,b,z) # || diff(x).*diff(z) - diff(x').*diff(z) ||
    return l2_norm(diff(a).*diff(z) , diff(b).*diff(z))
end

# """
# l2norm_strat_penalty: computes the Euclidean distance (l²-norm) between two vectors and adds
# the a proxy for the difference in the initial stratification from the corresponding temperature
# profiles, as approximated by the stratification at the bottom.
# """
# function l2norm_strat_penalty(a,b,z) # d(x,x') = || x - x' ||
#     α_proxy(x) = x[2] - x[1]
#     println(abs(α_proxy(a)-α_proxy(b)))
#     if abs(α_proxy(a)-α_proxy(b))>0.05
#         return l2_norm(a,b) + 0.0001
#     end
#     return l2_norm(a,b)
# end
