include("gp.jl")
include("distances.jl")

function get_me_true_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on true check for a single value of γ
    # computed on the scaled down (range [0,1] for temperature) profile values
    total_error = 0.0
    gpr_prediction = get_gpr_pred(𝒢, 𝒟; unscaled=false)
    n = 𝒟.Nt-2
    for i in 1:n
        exact    = 𝒟.y[i+1]
        predi    = gpr_prediction[i+1]
        total_error += euclidean_distance(exact, predi) # euclidean distance
    end
    return total_error / n
end

function get_me_greedy_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on greedy check
    # compares the direct model output and the target directly.
    # i.e. checks how well the model fits the data, NOT how well the temperature profiles calculated from the model output compares to the truth.
    # this means
    # - if the problem is a Sequential problem, it will compare the scaled (range [0,1] for temp profiles) values
    # - if the problem is a Residual problem, it will check the accuracy of the residual values computed by the model (which are used to compute the predicted T profiles), not the T profiles directly.

    total_error = 0.0
    # greedy check
    verification_set = 𝒟.verification_set
    n = length(verification_set)
    for j in 1:n
        test_index = verification_set[j]
        y_prediction = prediction(𝒟.x[test_index], 𝒢)
        error = euclidean_distance(y_prediction, 𝒟.y[test_index])
        total_error += error
    end
    return total_error / n
end
