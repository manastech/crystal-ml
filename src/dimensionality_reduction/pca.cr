# src/crystal_ml/dimensionality_reduction/pca.cr

require "../base" 
require "num" 

module CrystalML
  module DimensionalityReduction
    class PCA < UnsupervisedEstimator
      include Transformer
      property components : Tensor(Float64, CPU(Float64))
      property explained_variance : Array(Float64)
      property num_components : Int32

      def initialize(@num_components : Int32)
        @components = Tensor(Float64, CPU(Float64)).zeros([@num_components, @num_components])
        @explained_variance = [] of Float64
      end

      def fit(data : Tensor(Float64, CPU(Float64)))
        # Normalize the data (subtract the mean)
        means = data.mean(axis: 0)
        normalized_data = data - means
        
        # Compute the covariance matrix
        covariance_matrix = (normalized_data.transpose.matmul(normalized_data)) / (data.shape[0] - 1)

        # Find the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = covariance_matrix.eig

        # Sort the eigenvectors based on eigenvalues in descending order
        sorted_indices = eigenvalues.to_a.each_with_index.to_a.sort_by { |value, _| -value }.map { |pair| pair.last }

        sorted_eigenvectors = sorted_indices.map { |i| eigenvectors[i] }

        # Select the top 'num_components' eigenvectors
        i = 0
        sorted_eigenvectors[0...@num_components].each do |eigenvector|
          if i == 0
            @components = eigenvector.expand_dims(0)
          else
            @components = Num.concatenate(@components, eigenvector.expand_dims(0), axis: 0)
          end
          i += 1
        end

        # Calculate explained variance
        total_variance = eigenvalues.sum
        @explained_variance = sorted_indices[0...@num_components].map { |i| eigenvalues[i].value / total_variance }
      end

      def transform(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # Project data onto the principal components
        means = data.mean(axis: 0)
        normalized_data = data - means
        normalized_data.matmul(@components.transpose)
      end
    end
  end
end
