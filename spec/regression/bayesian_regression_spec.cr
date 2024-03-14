# spec/bayesian_regression_spec.cr

require "../spec_helper"
require "../../src/regression/bayesian_regression"

describe CrystalML::Regression::BayesianRegression do
  # Sample data for testing
  data_array = [
    [1.0, 1.5],
    [2.0, 2.1],
    [3.0, 3.2],
    [4.0, 4.3],
    [5.0, 5.4],
    [6.0, 6.5],
    [7.0, 7.2]
  ]
  target_array = [[3.0], [5.0], [7.0], [9.0], [11.0], [13.0], [15.0]]
  
  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  describe "#fit" do
    it "fits a Bayesian linear model without faling" do
      model = CrystalML::Regression::BayesianRegression.new
      model.fit(data_tensor, target_tensor)

      model.mean.should_not be_nil
      model.covariance.should_not be_nil
    end
  end

  describe "#predict" do
    it "makes predictions with mean using tensors" do
      model = CrystalML::Regression::BayesianRegression.new
      model.fit(data_tensor, target_tensor)

      mean_predictions = model.predict(data_tensor)

      mean_predictions.should be_a Tensor(Float64, CPU(Float64))
      mean_predictions.shape[0].should eq target_tensor.shape[0]

      mean_predictions.to_a.flatten.each_with_index do |mean_prediction, i|
        # Wide tolerance due to the nature of the model
        mean_prediction.should be_close target_array[i][0], 1
      end
    end
  end

  describe "#predict_variances" do
    it "makes predictions with variances using tensors" do
      model = CrystalML::Regression::BayesianRegression.new
      model.fit(data_tensor, target_tensor)

      variances = model.predict_variances(data_tensor)

      variances.should be_a Tensor(Float64, CPU(Float64))
      variances.shape[0].should eq target_tensor.shape[0]

      variances.to_a.flatten.each do |variance|
        variance.should be > 0
      end
    end
  end
end
