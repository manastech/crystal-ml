# src/crystal_ml/tree/decision_tree.cr

require "../base.cr"
require "num"

module CrystalML
  module Tree
    class Node
      property feature_index : Int32?
      property threshold : Float64?
      property left : Node?
      property right : Node?
      property value : Float64?
      
      def initialize(@feature_index : Int32? = nil,
        @threshold : Float64? = nil,
        @left : Node? = nil,
        @right : Node? = nil,
        @value : Float64? = nil)
      end

      def is_leaf? : Bool
        @feature_index.nil?
      end
    end
    
    abstract class DecisionTree < SupervisedEstimator
      property root : Node?
      property max_depth : Int32
      property min_size : Int32
      
      def initialize(@max_depth = 10, @min_size = 2)
        @root = nil
      end
      
      def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
        @root = build_tree(data, target, 0)
      end
      
      def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))
        predictions = Array(Float64).new
        
        data.each_axis(0) do |row|
          predictions << predict_row(@root, row)
        end
        
        Tensor(Float64, CPU(Float64)).from_array(predictions)
      end
      
      abstract def build_tree(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)), depth : Int32) : Node

      protected def predict_row(node : Node?, row : Tensor(Float64, CPU(Float64))) : Float64
        return Float64::INFINITY if node.nil? # TODO: Check if this is a reachable condition
        return node.value.not_nil! if node.feature_index.nil?
      
        if row[node.feature_index.not_nil!].value < node.threshold.not_nil!
          predict_row(node.left, row)
        else
          predict_row(node.right, row)
        end
      end
      
    end
  end
end
