# src/crystal_ml/clustering/k_means.cr

require "../base"
require "num" 
require "math"
require "random"

module CrystalML
  module Clustering
    class KMeans < Clusterer
      getter centroids : Tensor(Float64, CPU(Float64))
      property n_clusters : Int32
      property max_iters : Int32
      property tolerance : Float64

      # Constructor
      def initialize(@n_clusters : Int32 = 8, @max_iters : Int32 = 300, @tolerance : Float64 = 0.0001)
        @centroids = Tensor(Float64, CPU(Float64)).new([0]) # Placeholder initialization
      end

      # Fit the model to the data
      def fit(data : Tensor(Float64, CPU(Float64)))
        # Randomly initialize centroids
        @centroids = Tensor(Float64, CPU(Float64)).from_array(data.sample(@n_clusters))

        @max_iters.times do
          clusters = assign_clusters(data)
          new_centroids = update_centroids(data, clusters)

          handle_empty_clusters!(data, new_centroids, clusters)

          break if converged?(new_centroids)

          @centroids = new_centroids

        end
      end

      private def handle_empty_clusters!(data : Tensor(Float64, CPU(Float64)), new_centroids : Tensor(Float64, CPU(Float64)), clusters : Array(Int32))
        updated_centroids = [] of Array(Float64)
        
        index = 0
        new_centroids.each_axis(0) do |centroid|
          if clusters.count(index) == 0
            # Assign a random data point to the centroid
            random_row = data[Random.rand(data.shape[0])]
            updated_centroids << random_row.to_a
          else
            updated_centroids << centroid.to_a
          end
          index += 1
        end
      
        # Convert updated_centroids array back to a Tensor
        Tensor(Float64, CPU(Float64)).from_array(updated_centroids)
      end
      
      def predict(data : Tensor(Float64, CPU(Float64))) : Array(Int32)
        predictions = [] of Int32
        data.each_axis(0) do |point|
          predictions << closest_centroid(point)
        end
        predictions
      end
      
      private def assign_clusters(data : Tensor(Float64, CPU(Float64))) : Array(Int32)
        clusters = [] of Int32
        data.each_axis(0) do |point|
          clusters << closest_centroid(point)
        end
        clusters
      end

      private def update_centroids(data : Tensor(Float64, CPU(Float64)), clusters : Array(Int32)) : Tensor(Float64, CPU(Float64))
        ret = (0...@n_clusters).map do |cluster_index|
          cluster_points = [] of Array(Float64)
      
          data.each_axis(0).with_index do |point, index|
            break if index >= clusters.size
            cluster_points << point.to_a if clusters[index] == cluster_index
          end
      
          if cluster_points.empty?
            # Reassign to a random data point if the cluster is empty
            random_point = data[Random.rand(data.shape[0])].to_a
            cluster_points << random_point
          end
          
          mean(Tensor(Float64, CPU(Float64)).from_array(cluster_points)).to_a
        end
      
        Tensor(Float64, CPU(Float64)).from_array(ret)
      end
      

      private def converged?(new_centroids : Tensor(Float64, CPU(Float64))) : Bool
        i = 0
        @centroids.each_axis(0) do |centroid|
          return false if euclidean_distance(centroid, new_centroids[i]) > @tolerance
          i += 1
        end
        true
      end

      private def closest_centroid(point : Tensor(Float64, CPU(Float64))) : Int32
        closest_centroid_index = -1
        actual_centroid_index = 0
        closest_centroid_distance = Float64::INFINITY
        @centroids.each_axis(0) do |centroid|
          distance = euclidean_distance(point, centroid)
          if distance < closest_centroid_distance
            closest_centroid_index = actual_centroid_index
            closest_centroid_distance = distance
          end
          actual_centroid_index += 1
        end
        closest_centroid_index
      end

      private def euclidean_distance(point1 : Tensor(Float64, CPU(Float64)), point2 : Tensor(Float64, CPU(Float64))) : Float64
        Math.sqrt((point1 - point2).map { |x| x**2 }.sum)
      end

      private def mean(tensor : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        tensor.mean(axis: 0) # Assuming mean calculation along a specific axis is supported
      end
    end
  end
end
