require "../base"
require "./decision_tree_classifier.cr"
require "num" 

module CrystalML
  module Classification
    class RandomForestClassifier
      include Classifier
      property trees : Array(DecisionTreeClassifier)
      property n_trees : Int32
      property max_depth : Int32
      property min_size : Int32
      
      def initialize(@n_trees = 100, @max_depth = 10, @min_size = 2)
        @trees = Array(DecisionTreeClassifier).new
      end
      
      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        @trees = Array(DecisionTreeClassifier).new
        
        @n_trees.times do
          tree = build_tree(data, target)
          @trees << tree
        end
      end
      
      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        predictions = Array(Float64).new
        
        data.each_axis(0) do |row|
          row_predictions = Array(Float64).new
          
          @trees.each do |tree|
            row_predictions << tree.predict_row(tree.root, row)
          end
          predictions << Num.mean(Tensor(Float64, CPU(Float64)).from_array(row_predictions))
        end
        
        Tensor(Float64, CPU(Float64)).from_array(predictions)
      end

      def build_tree(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64))) : DecisionTreeClassifier
        tree = DecisionTreeClassifier.new(@max_depth, @min_size)
        tree.fit(data, target)
        tree
      end
    end
  end
end