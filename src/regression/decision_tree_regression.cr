# src/crystal_ml/regression/decision_tree_regressor.cr

require "../base.cr"
require "../utils/tree"
require "num" 

module CrystalML
  module Regression
    class DecisionTreeRegression < Tree::DecisionTree
      include Regressor
      
      def build_tree(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), depth : Int32) : Tree::Node
        if depth >= @max_depth || data.shape[0] <= @min_size
          return Tree::Node.new(value: mean(target))
        end
        
        best_feature, best_threshold = get_best_split(data, target)
        if best_feature.nil? || best_threshold.nil?
          return Tree::Node.new(value: mean(target))
        end

        left_data, left_target, right_data, right_target = split_dataset(data, target, best_feature, best_threshold)
        
        if left_data.shape[0] == 0 || right_data.shape[0] == 0
          return Tree::Node.new(value: mean(target))
        end
        
        left_node = build_tree(left_data, left_target, depth + 1)
        right_node = build_tree(right_data, right_target, depth + 1)
        
        Tree::Node.new(feature_index: best_feature, threshold: best_threshold, left: left_node, right: right_node)
      end
      
      private def mean(target : Tensor(Float64, CPU(Float64))) : Float64
        target.to_a.sum / target.size.to_f
      end
      
      private def get_best_split(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64))) : Tuple(Int32?, Float64?)
        best_variance_reduction = -Float64::INFINITY
        best_feature = nil
        best_threshold = nil
        
        num_features = data.shape[1]
        num_features.times do |feature_index|
          # Get unique thresholds for this feature
          thresholds = data[..., feature_index].to_a.uniq
      
          thresholds.each do |threshold|
            left_target, right_target = split_target_by_threshold(data[..., feature_index], target, threshold)
            
            # Skip variance reduction calculation if either split is empty
            if left_target.empty? || right_target.empty?
              next
            end
      
            variance_reduction = calculate_variance_reduction(target, left_target, right_target)
            if variance_reduction > best_variance_reduction
              best_variance_reduction = variance_reduction
              best_feature = feature_index
              best_threshold = threshold
            end
          end
        end
        {best_feature, best_threshold}
      end
      
      private def calculate_variance_reduction(total_target : Tensor(Float64, CPU(Float64)), left_target : Tensor(Float64, CPU(Float64)), right_target : Tensor(Float64, CPU(Float64))) : Float64
        total_variance = variance(total_target)
        left_variance = variance(left_target)
        right_variance = variance(right_target)
        total_size = total_target.size.to_f
        left_size = left_target.size.to_f
        right_size = right_target.size.to_f
        total_variance - (left_size / total_size) * left_variance - (right_size / total_size) * right_variance
      end
      
      private def variance(target : Tensor(Float64, CPU(Float64))) : Float64
        mean_val = mean(target)
        sum_of_squares = target.map { |val| (val - mean_val) ** 2 }.sum
        sum_of_squares / target.size.to_f
      end
      
      private def split_target_by_threshold(feature_column : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), threshold : Float64)
        # Initialize arrays to hold indices for left and right splits
        left_indices = [] of Int32
        right_indices = [] of Int32
        
        # Iterate over each element in the feature column
        feature_column.each_with_index do |value, index|
          if value < threshold
            left_indices << index
          else
            right_indices << index
          end
        end
        
        # Create tensors for the left and right targets using the collected indices
        left_target = Tensor(Float64, CPU(Float64)).new([left_indices.size, 1])
        right_target = Tensor(Float64, CPU(Float64)).new([right_indices.size, 1])
        
        left_indices.each_with_index do |idx, new_idx|
          left_target[new_idx][0] = target.to_a[idx]
        end
        
        right_indices.each_with_index do |idx, new_idx|
          right_target[new_idx][0] = target.to_a[idx]
        end
        
        [left_target, right_target]
      end

      private def split_dataset(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), feature_index : Int32, threshold : Float64)
        left_indices = [] of Int32
        right_indices = [] of Int32

        # Collect indices for data splitting
        index = 0
        data.each_axis(0) do |row|
          if row[feature_index].value < threshold
            left_indices << index
          else
            right_indices << index
          end
          index += 1
        end

        # Create new tensors for left and right data and targets
        left_data = Tensor(Float64, CPU(Float64)).new([left_indices.size, data.shape[1]])
        right_data = Tensor(Float64, CPU(Float64)).new([right_indices.size, data.shape[1]])
        left_target = Tensor(Float64, CPU(Float64)).new([left_indices.size, 1])
        right_target = Tensor(Float64, CPU(Float64)).new([right_indices.size, 1])

        # Fill the new tensors based on the indices collected
        left_indices.each_with_index do |data_index, new_index|
          left_data[new_index] = data[data_index]  # Assuming syntax for all columns in a row
          left_target[new_index][0] = target.to_a[data_index]
        end

        right_indices.each_with_index do |data_index, new_index|
          right_data[new_index] = data[data_index]  # Assuming syntax for all columns in a row
          right_target[new_index][0] = target.to_a[data_index]
        end

        [left_data, left_target, right_data, right_target]
      end
    end
  end
end
