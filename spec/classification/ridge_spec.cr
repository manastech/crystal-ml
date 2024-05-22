# spec/classification/ridge_classifier_spec.cr

require "../spec_helper"
require "../../src/classification/ridge"

describe CrystalML::Classification::RidgeClassifier do

  # Sample training data
  data_array = [
    [1.1, 2.2],
    [2.2, 1.1],
    [-1.2, -2.2],
    [-2.2, -1.2]
  ]
  # Sample target values for binary classification
  target_array = [1.0, 1.0, 0.0, 0.0]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  # Sample test data
  test_data_array = [
    [1.5, 2.5], # Expected to be classified as 1
    [-1.5, -2.5]  # Expected to be classified as 0
  ]
  test_data_tensor = Tensor(Float64, CPU(Float64)).from_array(test_data_array)

  describe "#fit and #predict" do
    it "correctly fits the model and predicts target values using tensors" do
      classifier = CrystalML::Classification::RidgeClassifier.new(alpha: 1.0)
      classifier.fit(data_tensor, target_tensor)
      
      predictions = classifier.predict(test_data_tensor).to_a
      predictions.should eq([1.0, 0.0])
    end
  end
end
