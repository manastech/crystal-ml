require "../base"
require "num"

module CrystalML
  module Regression
    class BayesianRegression < SupervisedEstimator
      include Regressor
      property mean : Tensor(Float64, CPU(Float64))
      property covariance : Tensor(Float64, CPU(Float64))
      
      # Hyperparameters for the prior distribution
      property alpha : Float64
      property beta : Float64
      
      def initialize(@alpha : Float64 = 1.0, @beta : Float64 = 1.0)
        @mean = Tensor(Float64, CPU(Float64)).zeros([1])
        @covariance = Tensor(Float64, CPU(Float64)).zeros([1])
      end
      
      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        # Manually augment the data with a column of ones for the intercept term
        ones = Tensor(Float64, CPU(Float64)).ones([data.shape[0], 1])
        x_matrix = Num.hstack([ones, data])
        x_transpose = x_matrix.transpose
        
        # Lambda Identity matrix for the prior covariance matrix
        lambda_i = Tensor(Float64, CPU(Float64)).eye(data.shape[1] + 1) * @alpha
        
        # Posterior covariance matrix = (lambda*I + beta*X^T*X)^-1
        @covariance = (lambda_i + x_transpose.matmul(x_matrix) * @beta).inv
        
        # If target is a 1D tensor, reshape it to a 2D tensor
        if target.shape.size == 1
          target = target.reshape([target.shape[0], 1])
        end

        # Posterior mean = beta * covariance * X^T * y
        @mean = @covariance.matmul(x_transpose).matmul(target) * @beta
      end
      
      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        ones = Tensor(Float64, CPU(Float64)).ones([data.shape[0], 1])
        x_matrix = Num.hstack([ones, data])

        x_matrix.matmul(@mean.not_nil!)
      end

      def predict_variances(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # TODO: predict variances should accept arrays and dataframes as well
        ones = Tensor(Float64, CPU(Float64)).ones([data.shape[0], 1])
        x_matrix = Num.hstack([ones, data])
        
        x_covariance = x_matrix.matmul(@covariance.not_nil!)
        x_covariance_xt = x_covariance.matmul(x_matrix.transpose)
        
        # Extract diagonal elements to get variances for each prediction and add 1/@beta
        predictive_variances = x_covariance_xt.diagonal.map { |var| var + 1.0 / @beta }

        predictive_variances
      end
    end
  end
end
