require "../base"
require "num"

module CrystalML
  module DimensionalityReduction
    class LDA < SupervisedEstimator
      include Transformer
      
      property components : Tensor(Float64, CPU(Float64))
      property num_components : Int32

      def initialize(@num_components : Int32)
        @components = Tensor(Float64, CPU(Float64)).zeros([@num_components, @num_components])
      end

      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        labels = target.to_a.map(&.to_i)
        unique_labels = labels.uniq
        overall_mean = data.mean(axis: 0)
      
        # Initialize within-class and between-class scatter matrices
        s_w = Tensor(Float64, CPU(Float64)).zeros([data.shape[1], data.shape[1]])
        s_b = Tensor(Float64, CPU(Float64)).zeros([data.shape[1], data.shape[1]])
      
        unique_labels.each do |label|
          data_class_indices = [] of Int32
          labels.each_with_index do |l, index| 
            if l == label
              data_class_indices << index
            end
          end

          data_class = [] of Array(Float64)
          row_n = 0
          data.each_axis(0) do |row| 
            if data_class_indices.includes?(row_n)
              data_class << row.to_a
            end
            row_n += 1
          end
          data_class = Tensor(Float64, CPU(Float64)).from_array (data_class)
          mean_class = data_class.mean(axis: 0)
        
          diff = data_class - mean_class
          s_w += diff.transpose.matmul(diff)
        
          mean_diff = mean_class - overall_mean
          s_b += mean_diff.expand_dims(1).matmul(mean_diff.expand_dims(0)) * data_class.shape[0]

        end

        # Solve the generalized eigenvalue problem for inv(s_w) * s_b
        eigenvalues, eigenvectors = (s_w.inv.matmul(s_b)).eig
      
        # Sort eigenvectors based on eigenvalues in descending order
        sorted_indices = eigenvalues.to_a.each_with_index.to_a.sort_by { |value, _| -value }.map { |pair| pair.last }

        # Select the top 'num_components' eigenvectors
        selected_indices = sorted_indices[0...@num_components]
        @components = Tensor(Float64, CPU(Float64)).zeros([eigenvectors.shape[0], selected_indices.size]).transpose
      
        selected_indices.each_with_index do |index, i|
          @components[i] = eigenvectors.transpose[index]
        end
        @components
      end
      
      def transform(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        data.matmul(@components.transpose)
      end
    end
  end
end
