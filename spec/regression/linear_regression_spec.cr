# spec/linear_regression_spec.cr

require "../spec_helper" 
require "../../src/regression/linear_regression"

describe CrystalML::Regression::LinearRegression do
  data_array = [
    [1.0, 1.5],
    [2.0, 2.1],
    [3.0, 3.2],
    [4.0, 4.3],
    [5.0, 5.4],
    [6.0, 6.5], 
    [7.0, 7.2]
  ]
  target_array = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
  
  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  data_df = Crysda.dataframe_of("feature1", "feature2").values(
    1.0, 1.5,
    2.0, 2.1,
    3.0, 3.2,
    4.0, 4.3,
    5.0, 5.4,
    6.0, 6.5, 
    7.0, 7.2
  )
  target_df = Crysda.dataframe_of("target").values(
    3.0,
    5.0,
    7.0,
    9.0,
    11.0,
    13.0,
    15.0
  )

  describe "#fit and #predict" do
    it "fits a linear model and makes predictions using tensors" do
      # Initialize and fit the model
      model = CrystalML::Regression::LinearRegression.new
      model.fit(data_tensor, target_tensor)

      # Check if coefficients are not nil and match expected pattern
      model.coefficients.should_not be_nil
      coefficients = model.coefficients.not_nil! 
      coefficients.shape[0].should eq 3 # including intercept

      expected_intercept = 1.0
      coefficients[0, 0].value.should be_close(expected_intercept, 0.001)

      # Make predictions and check their accuracy
      predictions = model.predict(data_tensor)
      predictions.shape[0].should eq target_tensor.shape[0]
      predictions.to_a.flatten.each_with_index do |prediction, i|
        prediction.should be_close(target_tensor[i].value, 0.001)
      end
    end

    it "fits a linear model and makes predictions using arrays" do
      model = CrystalML::Regression::LinearRegression.new
      model.fit(data_array, target_array)

      model.coefficients.should_not be_nil
      coefficients = model.coefficients.not_nil! 
      coefficients.shape[0].should eq 3 # including intercept

      expected_intercept = 1.0
      coefficients[0, 0].value.should be_close(expected_intercept, 0.001)

      predictions = model.predict(data_array)
      predictions.shape[0].should eq target_array.size
      predictions.to_a.flatten.each_with_index do |prediction, i|
        prediction.should be_close(target_array[i], 0.001)
      end
    end

    it "fits a linear model and makes predictions using a dataframe" do
      model = CrystalML::Regression::LinearRegression.new
      model.fit(data_df, target_df)

      # Check if coefficients are not nil and match expected pattern
      model.coefficients.should_not be_nil
      coefficients = model.coefficients.not_nil! 
      coefficients.shape[0].should eq 3 # including intercept

      expected_intercept = 1.0
      coefficients[0, 0].value.should be_close(expected_intercept, 0.001)

      predictions = model.predict(data_df)
      predictions.shape[0].should eq target_array.size
      predictions.to_a.flatten.each_with_index do |prediction, i|
        prediction.should be_close(target_array[i], 0.001)
      end
    end
  end
end