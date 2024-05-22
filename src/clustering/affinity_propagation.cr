require "../base"
require "num" 
require "math"

module CrystalML
  module Clustering
    class AffinityPropagation < UnsupervisedEstimator
      include Clusterer
      
      getter :exemplars, :responsibility_matrix, :availability_matrix
      property preference : Float64
      property damping : Float64
      property max_iters : Int32
      property convergence_iter : Int32
      property last_assignments : Array(Int32)?
      property stable_iterations : Int32 = 0

      def initialize(@preference : Float64 = -50.0, @damping : Float64 = 0.5, @max_iters : Int32 = 200, @convergence_iter : Int32 = 15)
        @exemplars = Array(Tensor(Float64, CPU(Float64))).new
        @responsibility_matrix = Tensor(Float64, CPU(Float64)).new([0]) # Placeholder
        @availability_matrix = Tensor(Float64, CPU(Float64)).new([0]) # Placeholder
      end

      def fit(data : Tensor(Float64, CPU(Float64)))
        s = similarity_matrix(data)
        @responsibility_matrix = Tensor(Float64, CPU(Float64)).zeros(s.shape)
        @availability_matrix = Tensor(Float64, CPU(Float64)).zeros(s.shape)

        @max_iters.times do |iteration|
          update_responsibility(s)
          update_availability()
          current_assignments = calculate_assignments
          break if check_convergence(current_assignments)
        end

        extract_exemplars(data)
      end

      def predict(data : Tensor(Float64, CPU(Float64))) : Array(Int32)
        predictions = Array(Int32).new(data.shape[0], 0)
      
        data.each_axis(0) do |point|
          closest_exemplar_index = -1
          smallest_distance = Float64::INFINITY
      
          @exemplars.each_with_index do |exemplar, index|
            distance = euclidean_distance(point, exemplar)
            if distance < smallest_distance
              smallest_distance = distance
              closest_exemplar_index = index
            end
          end
      
          predictions << closest_exemplar_index
        end
      
        predictions
      end

      private def similarity_matrix(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        n_points = data.shape[0]
        # Initialize the similarity matrix with zeros
        s = Tensor(Float64, CPU(Float64)).zeros([n_points, n_points])
      
        # Efficiently calculate the negative Euclidean distances (similarity)
        n_points.times do |i|
          n_points.times do |j|
            if i != j
              s[i][j] = -euclidean_distance(data[i], data[j])
            end
          end
        end
      
        # Set preferences
        preferences = s.flat.reject(&.== Float64::INFINITY).sort[(s.size * 0.5).to_i]
        n_points.times do |i|
          s[i, i] = preferences
        end
      
        s
      end

      private def update_responsibility(similarity : Tensor(Float64, CPU(Float64)))
        # Responsibility: r(i, k) = s(i, k) - max{a(i, k') + s(i, k')}
        # Note: This is a simplified version; actual implementation requires considering matrix operations for efficiency.
        r_new = @responsibility_matrix.dup
        n_points = similarity.shape[0]

        n_points.times do |i|
          n_points.times do |k|
            max_val = Float64::MIN
            n_points.times do |k_prime|
              next if k_prime == k
              new_possible_max = @availability_matrix[i][k_prime] + similarity[i][k_prime]
              if new_possible_max > max_val
                max_val = new_possible_max
              end
            end
            r_new[i, k] = similarity[i, k] - max_val
          end
        end

        @responsibility_matrix = r_new * @damping + @responsibility_matrix * (1 - @damping)
      end

      private def update_availability()
        # Availability: a(i, k) = min{0, r(k, k) + sum{max{0, r(i', k)}}}
        # Note: Simplified for clarity. Use matrix operations for real implementation.
        a_new = @availability_matrix.dup
        n_points = @responsibility_matrix.shape[0]

        n_points.times do |k|
          n_points.times do |i|
            if i == k
              sum_val = 0.0
              n_points.times { |i_prime| sum_val += [@responsibility_matrix[i_prime][k].value, 0.0].max if i_prime != k }
              a_new[i][k] = sum_val
            else
              sum_val = 0.0
              n_points.times { |i_prime| sum_val += [@responsibility_matrix[i_prime][k].value, 0.0].max unless i_prime == k || i_prime == i }
              a_new[i][k] = [0, @responsibility_matrix[k][k].value + sum_val].min
            end
          end
        end

        @availability_matrix = a_new * @damping + @availability_matrix * (1 - @damping)
      end

      private def check_convergence(assignments : Array(Int32))
        if last_assignments.nil? || last_assignments != assignments
          self.stable_iterations = 0
          self.last_assignments = assignments
        else
          self.stable_iterations += 1
        end
      
        self.stable_iterations >= convergence_iter
      end

      private def calculate_assignments
        n_points = @availability_matrix.shape[0]
        assignments = Array(Int32).new(n_points, 0)
      
        n_points.times do |i|
          max_index = -1
          max_value = Float64::INFINITY * -1
          
          n_points.times do |k|
            value = @availability_matrix[i][k] + @responsibility_matrix[i][k]
            if value > max_value
              max_value = value
              max_index = k
            end
          end
          assignments[i] = max_index
        end
      
        assignments
      end
      
      private def extract_exemplars(data : Tensor(Float64, CPU(Float64)))
        assignments = calculate_assignments
        @exemplars = Array(Tensor(Float64, CPU(Float64))).new
      
        assignments.uniq.each do |exemplar_index|
          @exemplars << data[exemplar_index]
        end
      
        @exemplars
      end

      private def euclidean_distance(point1 : Tensor(Float64, CPU(Float64)), point2 : Tensor(Float64, CPU(Float64))) : Float64
        Math.sqrt((point1 - point2).map { |x| x**2 }.sum)
      end
    end
  end
end
