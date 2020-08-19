include("GP.jl")
include("distances.jl")

function get_me_true_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on true check for a single value of γ
    # computed on the scaled down (range [0,1]) profile values
    total_error = 0.0
    gpr_prediction = get_gpr_pred(𝒢, 𝒟; unscaled=false)
    n = 𝒟.Nt-2
    for i in 1:n
        exact    = 𝒟.y[i+1]
        predi    = gpr_prediction[i+1]
        total_error += l2_norm(exact, predi) # euclidean distance
    end
    return total_error / n
end

function get_me_greedy_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on greedy check for a single value of γ
    # computed on the scaled down (range [0,1]) profile values
    total_error = 0.0
    n = length(𝒟.verification_set)
    # greedy check
    verification_set = 𝒟.verification_set
    for j in 1:n
        test_index = verification_set[j]
        y_prediction = prediction(𝒟.x[test_index], 𝒢)
        error = l2_norm(y_prediction, 𝒟.y[test_index])
        total_error += error
    end
    return total_error / n
end
