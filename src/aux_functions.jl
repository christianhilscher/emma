# Saving auxillary function which I need over multiple files

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

    return Y
end

function dp3(x::Matrix, errors::Matrix)

    Y = 4 .* (x[:, 1] .- 2 + 8 .* x[:, 2] .- 8 .* x[:,2].^2).^2 + (3 .- 4 .* x[:,2]).^2 + 16 .* (x[:, 3] .+ 1).^(0.5) .* (2 .* x[:,3] .- 1).^2 + errors

    return Y
end

function dp8(x::Matrix, errors::Matrix)

    Y = 4 .* (x[:, 1] .- 2 + 8 .* x[:, 2] .- 8 .* x[:,2].^2).^2 + (3 .- 4 .* x[:,2]).^2 + 16 .* (x[:, 3] .+ 1).^(0.5) .* (2 .* x[:,3] .- 1).^2 .+ sum(x[:, 4:8] .* (1 .+ log.(sum(x[:,1:3], dims=2))), dims=2) + errors

    return Y
end

function robot(x::Matrix, errors::Matrix)

    x[:, 1:4] = x[:, 1:4] .* 2 .* π     # Scaling inputs

    u = sum(x[:, 5:8] .* cos.(sum(x[:, 1:4], dims=2)) ,dims=2)
    v = sum(x[:, 5:8] .* sin.(sum(x[:, 1:4], dims=2)) ,dims=2)

    Y = (u.^2 .+ v.^2).^(0.5) + errors
    return Y
end

function sine_easy(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1]) .+ errors

    return Y
end

function make_data(n, d, func, σ)

    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    d = Normal(0, σ)
    td = truncated(d, -Inf, Inf)

    errors_train = rand(td, n, 1)
    errors_test = zeros(n, 1)

    if func=="friedman"
        y_train = friedman(x_train, errors_train)
        y_test = friedman(x_test, errors_test)
    elseif func=="sine_easy"
        y_train = sine_easy(x_train, errors_train)
        y_test = sine_easy(x_test, errors_test)
    elseif func=="dp3"
        y_train = dp3(x_train, errors_train)
        y_test = dp3(x_test, errors_test)
    elseif func=="dp8"
        y_train = dp8(x_train, errors_train)
        y_test = dp8(x_test, errors_test)
    elseif func=="robot"
        y_train = robot(x_train, errors_train)
        y_test = robot(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end


    return x_train, x_test, y_train, y_test
end

function get_mse(pred, y)
    bias = abs(mean(pred .- y))
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    return bias, variance, mse
end

function compare_models(rf1::RFR, rf2::RFR, x_test::Matrix, y_test::Matrix)


    pred1 = predict(rf1, x_test)
    pred2 = predict(rf2, x_test)


    println("Model 1: ", round.(get_mse(pred1, y_test), digits=5))
    println("Model 2: ", round.(get_mse(pred2, y_test), digits=5), "\n")
    println("Change 1 to 2: ", round.(get_mse(pred1, y_test)./get_mse(pred2, y_test), digits=5))
end