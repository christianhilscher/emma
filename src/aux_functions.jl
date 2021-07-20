# Saving auxillary function which I need over multiple files

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

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