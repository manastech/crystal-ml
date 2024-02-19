# src/crystal_ml/regression/linear_regression.cr

require "../base" 
require "num" 

module CrystalML
  module Regression
    class LinearRegression < Regressor
      property coefficients : Tensor(Float64, CPU(Float64))?

      def initialize
        @coefficients = nil
      end

      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        # Manually augment the data with a column of ones for the intercept term
        ones = Tensor(Float64, CPU(Float64)).ones([data.shape[0], 1])
        x_matrix = Num.hstack([ones, data])
      
        # Transpose the target tensor if necessary to ensure proper dimensions
        y_data = if target.shape.size == 1
          Tensor(Float64, CPU(Float64)).from_array(target.to_a.map { |val| [val] })
        else
          target
        end
        
        x_transpose = x_matrix.transpose
      
        # Solve the normal equation X^T * X * w = X^T * y
        xtx = x_transpose.matmul(x_matrix)
        xty = x_transpose.matmul(y_data)

        # Check if xtx is square and invertible
        if xtx.shape[0] == xtx.shape[1]
          @coefficients = xtx.inv.matmul(xty)
        else
          raise "XTX is not square: shape is #{xtx.shape}"
        end
      end

      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        raise "Model not fitted" unless @coefficients

        ones = Tensor(Float64, CPU(Float64)).ones([data.shape[0], 1])
        x_matrix = Num.hstack([ones, data])

        predictions = x_matrix.matmul(@coefficients.not_nil!)
        predictions
      end
    end
  end
end