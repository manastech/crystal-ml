# src/crystal_ml/classification/naive_bayes.cr

require "../base.cr"
require "num" 

module CrystalML
  module Classification
    class NaiveBayesClassifier < Classifier
      property means : Hash(Float64, Tensor(Float64, CPU(Float64)))
      property variances : Hash(Float64, Tensor(Float64, CPU(Float64)))

      def initialize
        @means = {} of Float64 => Tensor(Float64, CPU(Float64))
        @variances = {} of Float64 => Tensor(Float64, CPU(Float64))
      end

      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        grouped_data = group_data_by_class(data, target)

        grouped_data.each do |class_val, class_data|
          @means[class_val] = compute_means(class_data)
          @variances[class_val] = compute_variances(class_data, @means[class_val])
        end
      end

      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        predictions = Array(Float64).new

        data.each_axis(0) do |row|
          class_probabilities = @means.keys.map do |class_val|
            [class_val, calculate_probability(row, class_val)]
          end.to_h
          predictions << class_probabilities.max_by { |_, prob| prob }.first
        end

        Tensor(Float64, CPU(Float64)).from_array(predictions)
      end

      private def compute_means(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        data.mean(axis: 0)
      end

      private def compute_variances(data : Tensor(Float64, CPU(Float64)), means : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        differences = data - means
        squared_differences = differences ** 2
        squared_differences.mean(axis: 0)
      end

      private def group_data_by_class(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64))) : Hash(Float64, Tensor(Float64, CPU(Float64)))
        grouped_data = {} of Float64 => Tensor(Float64, CPU(Float64))

        target.each_with_index do |class_val, index|
          if  grouped_data.has_key?(class_val)
            grouped_data[class_val] = Num.concatenate([grouped_data[class_val],data[index].expand_dims(0)], 0)
          else
            grouped_data[class_val] = data[index].expand_dims(0)
          end
        end

        grouped_data
      end

      private def calculate_probability(features : Tensor(Float64, CPU(Float64)), class_val : Float64) : Float64
        mean = @means[class_val]
        variance = @variances[class_val]
        g_prob = gaussian_probability(features, mean, variance)
        g_prob = g_prob.map { |v| v.nan? ? 1.0 : v }
        g_prob.prod
      end

      private def gaussian_probability(x : Tensor(Float64, CPU(Float64)), mean : Tensor(Float64, CPU(Float64)), variance : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        exp_part = ((x - mean) ** 2) / (2 * variance)
        sqrt_variance = variance.map { |v| Math.sqrt(v) } # Element-wise square root
        exp_part = exp_part.map { |v| Math.exp(-v) }      # Element-wise exponential
        (1 / (Math.sqrt(2 * Math::PI) * sqrt_variance)) * exp_part
      end
    end
  end
end
