require "../base"
require "./decision_tree_regression.cr"
require "num"

module CrystalML
  module Regression
    class RandomForestRegression
      include Regressor
      property trees : Array(DecisionTreeRegression)
      property n_trees : Int32
      property max_depth : Int32
      property min_size : Int32
      
      def initialize(@n_trees = 100, @max_depth = 10, @min_size = 2)
        @trees = Array(DecisionTreeRegression).new
      end
      
      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        @trees.clear

        @n_trees.times do
          sample_data, sample_target = bootstrap_sample(data, target)
          tree = build_tree(sample_data, sample_target)
          @trees << tree
        end
      end
      
      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        predictions = Array(Array(Float64)).new(data.shape[0]) { Array(Float64).new(@n_trees, 0.0) }

        i=0
        data.each_axis(0) do |row|
          @trees.each_with_index do |tree, j|
            predictions[i][j] = tree.predict_row(tree.root, row)
          end
          i+=1
        end

        averaged_predictions = predictions.map do |tree_preds|
          Num.mean(Tensor(Float64, CPU(Float64)).from_array(tree_preds))
        end

        Tensor(Float64, CPU(Float64)).from_array(averaged_predictions)
      end

      private def build_tree(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64))) : DecisionTreeRegression
        tree = DecisionTreeRegression.new(@max_depth, @min_size)
        tree.fit(data, target)
        tree
      end

      private def bootstrap_sample(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        sample_indices = Array(Int32).new(data.shape[0]) { Random.rand(0..data.shape[0] - 1) }
        sample_data = sample_indices.map { |i| data[i,].to_a }
        sample_target = sample_indices.map { |i| target[i].value }

        [
          Tensor(Float64, CPU(Float64)).from_array(sample_data),
          Tensor(Float64, CPU(Float64)).from_array(sample_target)
        ]
      end
    end
  end
end
