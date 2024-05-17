# spec/regression/decision_tree_regressor_spec.cr

require "../spec_helper"
require "../../src/regression/random_forest_regression"

describe CrystalML::Regression::RandomForestRegression do
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
      regressor = CrystalML::Regression::RandomForestRegression.new()
      regressor.fit(data_tensor, target_tensor)

      #regressor.print_tree()

    end
  end

  describe "#predict" do
    it "correctly predicts the target value of given data points" do
      regressor = CrystalML::Regression::RandomForestRegression.new()
      regressor.fit(data_tensor, target_tensor)

      test_data = [
        [1.1, 2.5],
        [2.1, 3.0],
        [3.2, 4.2],
        [4.1, 5.2],
        [5.2, 6.0]
      ]
      test_data_tensor = Tensor(Float64, CPU(Float64)).from_array(test_data)
      
      predictions = regressor.predict(test_data_tensor).to_a

      target_array.each_with_index do |target, i|
        predictions[i].should be_close(target, 1.0)
      end
    end
  end
end
