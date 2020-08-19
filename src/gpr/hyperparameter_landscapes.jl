include("gaussian_process.jl")
include("errors.jl")

function plot_landscapes_compare_error_metrics(k::Int64, 𝒟::ProfileData, d, γs)
    # Compare mean log marginal likelihood with
    #    mean error on greedy check and
    #    mean error on true check

    mlls = zeros(length(γs)) # mean log marginal likelihood
    mes  = zeros(length(γs)) # mean error (greedy check)
    mets  = zeros(length(γs)) # mean error (true check)

    for i in 1:length(γs)

        kernel = get_kernel(k, γs[i], 0.0)
        𝒢 = get_gp(𝒟, kernel, d)

        # -----compute mll loss----
        mlls[i] = -1*mean_log_marginal_loss(𝒟.y_train, 𝒢, add_constant=false)

        # -----compute mean error for greedy check (same as in plot log error)----
        mes[i] = get_me_greedy_check(𝒢, 𝒟)

        # -----compute mean error for true check----
        mets[i] = get_me_true_check(𝒢, 𝒟)

    end

    ylims = ( minimum([minimum(mets), minimum(mes)]) , maximum([maximum(mets), maximum(mes)]) )

    mll_plot = plot(γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
    vline!([γs[argmin(mlls)]])
    mes_plot  = plot(γs, mes,  xlabel="log(γ)", title="ME on greedy check, min = $(round(minimum(mes);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mes)]])
    met_plot  = plot(γs, mets,  xlabel="log(γ)", title="ME on true check, min = $(round(minimum(mets);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mets)]])

    return plot(mll_plot, mes_plot, met_plot, layout = @layout [a ; b; c])
end


function plot_landscapes_compare_files_me(filenames, k::Int64, d, γs, v_str; N=4)
    # visual comparison of the mean error on true check for every file in filenames

    function get_me(filename)
        𝒟 = construct_profile_data(file, v_str, 16; N=N)

        mes  = zeros(length(γs))
        for i in 1:length(γs)
            kernel = get_kernel(k, γs[i], 0.0)
            𝒢 = get_gp(𝒟, kernel, d);
            mes[i] = get_me_true_check(𝒢, 𝒟)
        end

        return mes
    end

    results = Dict(file => get_me(file) for file in filenames])

    # put all the data into one array for plotting
    for r in results
        all = hcat(all, r[file])
    end

    layout = (length(filenames), 1)
    ylims = (minimum(all),maximum(all))

    # minimizing γ values
    argmin_logγ = vcat([γs[argmin(results[file])]
                for file in filenames])

    titles = ["$(file), log(γ)=$(argmin_logγ[i]), min = $(round(minimum(results[filenames[i]]);digits=5))"
             for i in eachindex(filenames)]

    p = plot(γs, xlabel="log(γ)", ylabel="ME, true check", title=titles, legend=false, yscale=:log10, ylims=ylims, layout=layout)  # 1D plot: mean error vs. γ

    vline!(argmin_γ')

    return p
end

function plot_error_histogram(𝒢::GP, 𝒟::ProfileData, time_index)
    # mean error for true check
    gpr_prediction = get_gpr_pred(𝒢, 𝒟)
    gpr_error = zeros(𝒟.Nt-2)
    for i in 1:𝒟.Nt-2
        exact    = data.y[i+1]
        predi    = gpr_prediction[i+1]
        gpr_error[i] = l2_norm(exact, predi) # euclidean distance
    end
    mean_error = sum(gpr_error)/(𝒟.Nt-2)

    error_plot_log = histogram(log.(gpr_error), title = "log(error) at each timestep of the full evolution", xlabel="log(Error)", ylabel="Frequency",ylims=(0,250), label="frequency")
    vline!([log(mean_error)], line = (4, :dash, 0.8), label="mean error")
    vline!([log(gpr_error[time_index])], line = (1, :solid, 0.6), label="error at t=$(time_index)")
end

function get_min_gamma(k::Int64, data::ProfileData, d, γs)
    # returns the gamma value that minimizes the mean error on the true check
    # - only tests the gamma values listed in the γs parameter

    mets  = zeros(length(γs)) # mean error for each gamma (true check)
    for (i, logγ) in enumerate(γs)

        kernel = get_kernel(k, logγ, 0.0)
        𝒢 = get_gp(data, kernel, d);

        # -----compute mean error for true check----
        total_error = 0.0
        gpr_prediction = get_gpr_pred(𝒢, data)
        for q in 1:data.Nt-2
            exact        = data.y[q+1]
            predi        = gpr_prediction[q+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(data.Nt-2)
    end

    i = argmin(mets)
    min_logγ = γs[i]
    min_error = mets[i]

    return (min_logγ, min_error)
end

function get_min_gamma_alpha(k::Int64, data::ProfileData, d, γs)
    # returns the gamma value that minimizes the mean error on the true check
    # only tests the gamma values listed in γs parameter

    mets  = zeros(length(γs*αs)) # mean error for each gamma (true check)

    for i in eachindex(γs), j in eachindex(αs)

        kernel = RationalQuadraticI(γ[i], 0.0, αs[j])
        𝒢 = get_gp(data, kernel, d);

        # -----compute mean error for true check----
        total_error = 0.0
        gpr_prediction = get_gpr_pred(𝒢, data)
        for index in 1:data.Nt-2
            exact        = data.y[index+1]
            predi        = gpr_prediction[index+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(data.Nt-2)
    end
    end

    i = argmin(mets)
    min_gamma = γs[i]
    min_error = mets[i]

    return (min_gamma, min_error)
end
