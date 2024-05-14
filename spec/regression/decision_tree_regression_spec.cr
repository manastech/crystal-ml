# spec/regression/decision_tree_regressor_spec.cr

require "../spec_helper"
require "../../src/regression/decision_tree_regression"

describe CrystalML::Regression::DecisionTreeRegression do
  data_array = [
    [1.1, 2.5],
    [2.1, 3.0],
    [3.2, 4.2],
    [4.1, 5.2],
    [5.2, 6.0]
  ]
  target_array = [0.2, 1.3, 2.4, 3.5, 4.8] # Continuous values for regression

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  describe "#fit" do
    it "correctly builds the regression tree from training data" do
      regressor = CrystalML::Regression::DecisionTreeRegression.new(max_depth: 3, min_size: 1)
      regressor.fit(data_tensor, target_tensor)

      #regressor.print_tree()

    end
  end

  describe "#predict" do
    it "correctly predicts the target value of given data points" do
      regressor = CrystalML::Regression::DecisionTreeRegression.new(max_depth: 3, min_size: 1)
      regressor.fit(data_tensor, target_tensor)

      test_data = [
        [1.1, 2.5], # Expected target close to 0.2
        [4.1, 5.2]  # Expected target close to 3.5
      ]
      test_data_tensor = Tensor(Float64, CPU(Float64)).from_array(test_data)
      
      predictions = regressor.predict(test_data_tensor).to_a

      predictions[0].should be_close(0.2, 0.1)
      predictions[1].should be_close(3.5, 0.1)
    end
  end
end
