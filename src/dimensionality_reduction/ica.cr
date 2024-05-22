require "../base"
require "num"

module CrystalML
  module DimensionalityReduction
    class ICA < UnsupervisedEstimator
      include Transformer
      property components : Tensor(Float64, CPU(Float64))
      property mixing_matrix : Tensor(Float64, CPU(Float64))
      property num_components : Int32

      def initialize(@num_components : Int32)
        @components = Tensor(Float64, CPU(Float64)).zeros([@num_components, @num_components])
        @mixing_matrix = Tensor(Float64, CPU(Float64)).zeros([@num_components, @num_components])
      end

      def fit(data : Tensor(Float64, CPU(Float64)))
        # Center and whiten data
        means = data.mean(axis: 0)
        puts "means: #{means}"
        centered_data = data - means
        puts "centered_data: #{centered_data}"
        whitened_data = whiten(centered_data)
        puts "whitened_data: #{whitened_data}"

        # Initialize weights randomly
        weights = Tensor.random(0.0...1.0, [@num_components, whitened_data.shape[1]])
        puts "weights: #{weights}"
        max_iterations = 10
        tolerance = 1e-4

        max_iterations.times do |iteration|
          # Update weights using the FastICA algorithm
          w_plus = update_weights(whitened_data, weights)
          puts "w_plus: #{w_plus}"

          # Check for convergence
          lim = (w_plus.matmul(weights.transpose)).diagonal.map { |x| (x.abs - 1).abs }.max
          puts "lim: #{lim}"
          break if lim < tolerance

          weights = w_plus
        end

        @components = weights
        @mixing_matrix = weights.inv
      end

      private def create_diagonal(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # Create a diagonal matrix from the input data
        diagonal = Tensor(Float64, CPU(Float64)).zeros([data.shape[0], data.shape[0]])
        data.shape[0].times do |i|
          diagonal[i, i] = data[i]
        end
        diagonal
      end

      private def whiten(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # Compute the covariance matrix of the data
        covariance_matrix = (data.transpose.matmul(data)) / (data.shape[0] - 1)
      
        # Eigenvalue decomposition of the covariance matrix
        eigenvalues, eigenvectors = covariance_matrix.eig
      
        # Compute the inverse square root of the eigenvalues matrix
        diag_inv_sqrt_eigenvalues = create_diagonal(eigenvalues.map { |val| 1.0 / Math.sqrt(val) })

        # Whitening transformation: E*D^(-1/2)*E^T * data^T
        whitening_matrix = eigenvectors.matmul(diag_inv_sqrt_eigenvalues).matmul(eigenvectors.transpose)
      
        # Apply the whitening transformation to the data
        whitened_data = whitening_matrix.matmul(data.transpose)
      
        # Return the whitened data transposed back into original shape
        whitened_data.transpose
      end
      

      private def update_weights(data : Tensor(Float64, CPU(Float64)), weights : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # Define the non-linearity (here, using g(x) = tanh(x) for simplicity)
        g = ->(x : Float64) { Math.tanh(x) }
        g_derivative = ->(x : Float64) { 1.0 - (Math.tanh(x) ** 2) }
      
        # Compute the dot product of data and weights
        weighted_data = data.matmul(weights.transpose)
      
        # Apply the non-linear function g to the weighted data
        gw = weighted_data.map(&g)
      
        # Compute the derivative of the non-linear function
        g_w_derivative = weighted_data.map(&g_derivative)
        
        puts "**********************************"
        puts "data: #{data}"
        puts "weighted_data: #{weighted_data}"
        puts "gw: #{gw}"
        puts "g_w_derivative: #{g_w_derivative}"
        puts "g_w_derivative.mean: #{g_w_derivative.mean(axis: 0)}"
        puts "gw.transpose.matmul(data): #{gw.transpose.matmul(data)}"
        puts "substraction: #{gw.transpose.matmul(data) - g_w_derivative.mean(axis: 0)}"
        puts "**********************************"
        

        # Estimate the new weights
        new_weights = (gw.transpose.matmul(data) - g_w_derivative.mean(axis: 0)).matmul(weights) / data.shape[0]
      
        # Orthogonalize the new weights using symmetric decorrelation
        wwt = new_weights.matmul(new_weights.transpose)
        e, v = wwt.eig # Eigenvalue decomposition
        d_inv_sqrt = create_diagonal(e.map { |val| 1.0 / Math.sqrt(val) })
        new_weights = (v.matmul(d_inv_sqrt).matmul(v.transpose)).matmul(new_weights)
      
        new_weights
      end

      def transform(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        # Use the unmixing matrix to transform data to independent components
        centered_data = data - data.mean(axis: 0)
        whitened_data = whiten(centered_data)
        whitened_data.matmul(@components.transpose)
      end
    end
  end
end
