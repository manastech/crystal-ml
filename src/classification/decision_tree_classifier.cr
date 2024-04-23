# src/crystal_ml/classification/decision_tree_classifier.cr

require "../base.cr"
require "../utils/tree"
require "num" 

module CrystalML
  module Classification
    class DecisionTreeClassifier < Tree::DecisionTree
      include Classifier

      def build_tree(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), depth : Int32) : Tree::Node
        if depth >= @max_depth || data.shape[0] <= @min_size
          return Tree::Node.new(value: most_common_class(target))
        end
        
        best_feature, best_threshold = get_best_split(data, target)
        if best_feature.nil? || best_threshold.nil?
          return Tree::Node.new(value: most_common_class(target))
        end
        
        left_data, left_target, right_data, right_target = split_dataset(data, target, best_feature, best_threshold)

        if left_data.shape[0] == 0 || right_data.shape[0] == 0
          return Tree::Node.new(value: most_common_class(target))
        end
        
        left_node = build_tree(left_data, left_target, depth + 1)
        right_node = build_tree(right_data, right_target, depth + 1)
        
        Tree::Node.new(feature_index: best_feature, threshold: best_threshold, left: left_node, right: right_node)
      end

      def print_tree()
        print_tree_recursive(@root, 0)
      end

      def print_tree_recursive(node : Tree::Node | Nil, depth : Int32)
        return if node.nil?
        if node.is_leaf?
          puts "#{depth}Predict: #{node.value}"
          return
        end

        puts "#{depth}Feature #{node.feature_index} < #{node.threshold}"
        print_tree_recursive(node.left, depth + 1)
        puts "#{depth}Feature #{node.feature_index} >= #{node.threshold}"
        print_tree_recursive(node.right, depth + 1)
      end
      
      private def get_best_split(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64))) : Tuple(Int32?, Float64?)
        best_gini = Float64::INFINITY
        best_feature = nil
        best_threshold = nil
        
        num_features = data.shape[1]
        num_features.times do |feature_index|
          data[..., feature_index].to_a.uniq.each do |threshold|
            left_target, right_target = split_target_by_threshold(data[..., feature_index], target, threshold)
            gini = calculate_weighted_gini(left_target, right_target)
            if gini < best_gini
              best_gini = gini
              best_feature = feature_index
              best_threshold = threshold
            end
          end
        end
        {best_feature, best_threshold}
      end
      
      private def split_dataset(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), feature_index : Int32, threshold : Float64)
        left_indices = Array(Int32).new
        right_indices = Array(Int32).new

        index = 0
        data.each_axis(0) do |row|
          if row[feature_index].value < threshold
            left_indices << index
          else
            right_indices << index
          end
          index += 1
        end
      
        left_data = Tensor(Float64, CPU(Float64)).new([left_indices.size, data.shape[1]])
        right_data = Tensor(Float64, CPU(Float64)).new([right_indices.size, data.shape[1]])
        left_target = Tensor(Float64, CPU(Float64)).new([left_indices.size])
        right_target = Tensor(Float64, CPU(Float64)).new([right_indices.size])
      
        left_indices.each_with_index do |data_index, new_index|
          left_data[new_index] = data[data_index] 
          left_target[new_index] = target[data_index]
        end
      
        right_indices.each_with_index do |data_index, new_index|
          right_data[new_index] = data[data_index] 
          right_target[new_index] = target[data_index]
        end
      
        [left_data, left_target, right_data, right_target]
      end      
      
      private def split_target_by_threshold(feature_column : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), threshold : Float64)
        left_indices = Array(Int32).new
        right_indices = Array(Int32).new
      
        feature_column.each_with_index do |value, index|
          if value < threshold
            left_indices << index
          else
            right_indices << index
          end
        end
      
        left_target = Tensor(Float64, CPU(Float64)).new([left_indices.size, 1]) 
        right_target = Tensor(Float64, CPU(Float64)).new([right_indices.size, 1])
      
        left_indices.each_with_index do |index, new_index|
          left_target[new_index][0] = target[index]
        end
      
        right_indices.each_with_index do |index, new_index|
          right_target[new_index][0] = target[index]
        end
      
        [left_target, right_target]
      end      
      
      private def most_common_class(target : Tensor(Float64, CPU(Float64))) : Float64
        class_counts = target.to_a.tally
        most_common = class_counts.max_by { |_, count| count }.first
        most_common
      end
      
      private def calculate_weighted_gini(left_target : Tensor(Float64, CPU(Float64)), right_target : Tensor(Float64, CPU(Float64))) : Float64
        total_size = left_target.size.to_f + right_target.size.to_f
        left_gini = calculate_gini_impurity(left_target)
        right_gini = calculate_gini_impurity(right_target)
        (left_target.size.to_f / total_size) * left_gini + (right_target.size.to_f / total_size) * right_gini
      end
      
      private def eq_and_mean(target : Tensor(Float64, CPU(Float64)), class_val : Float64) : Float64
        matches = 0
        total = target.size
      
        target.each do |value|
          matches += 1 if value == class_val
        end
      
        matches.to_f / total
      end      

      private def calculate_gini_impurity(target : Tensor(Float64, CPU(Float64))) : Float64
        impurity = 1.0
        classes = target.to_a.uniq
        classes.each do |class_val|
          prob = eq_and_mean(target, class_val)
          impurity -= prob ** 2
        end
        impurity
      end
    end
  end
end