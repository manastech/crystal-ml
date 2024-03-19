# src/crystal_ml/classification/ridge.cr

require "../base.cr"
require "num"

module CrystalML
  module Classification
    class RidgeClassifier < SupervisedEstimator
      include Classifier
        
      property coef_ : Tensor(Float64, CPU(Float64))
      property intercept_ : Float64
      property alpha : Float64

      def initialize(@alpha : Float64 = 1.0)
        @coef_ = Tensor(Float64, CPU(Float64)).zeros([1])
        @intercept_ = 0.0
      end

      def fit(x : Tensor(Float64, CPU(Float64)), y : Tensor(Float64, CPU(Float64)), n_iter : Int32 = 10, tol : Float64 = 1e-3)
        n_samples, n_features = x.shape
        x_bias = Num.concatenate([x, Tensor(Float64, CPU(Float64)).ones([n_samples, 1])], 1)
        w = Tensor(Float64, CPU(Float64)).ones([n_features + 1, 1])

        n_iter.times do |iter|
          y_pred = x_bias.matmul(w)

          grad = x_bias.transpose.matmul(y_pred - y) + @alpha * w
          w -= grad * tol

          break if grad.map { |x| x.abs }.max < tol
        end

        @coef_ = w[0...-1]
        @intercept_ = w[-1][0].value
      end

      def predict(x : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        y_pred = x.matmul(@coef_) + @intercept_

        ret = [] of Float64
        y_pred.each_axis(0) do |row|
          ret << ((row[0].value) > 0.0 ? 1.0 : 0.0)
        end
        
        Tensor(Float64, CPU(Float64)).from_array(ret)
      end
    end
  end
end
