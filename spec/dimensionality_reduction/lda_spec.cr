require "../spec_helper"
require "../../src/dimensionality_reduction/lda"

describe CrystalML::DimensionalityReduction::LDA do
  data_array = [
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
  ]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)

  data_df = Crysda.dataframe_of("feature1", "feature2").values(
    2.5, 2.4,
    0.5, 0.7,
    2.2, 2.9,
    1.9, 2.2,
    3.1, 3.0,
    2.3, 2.7,
    2.0, 1.6,
    1.0, 1.1,
    1.5, 1.6,
    1.1, 0.9)

  labels = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)

  expected_components = Tensor(Float64, CPU(Float64)).from_array([[0.755689, 0.404834]])

  expected_transformed_data = Tensor(Float64, CPU(Float64)).from_array([
    [2.86083 ],
    [0.661229],
    [2.83654 ],
    [2.32644 ],
    [3.55714 ],
    [2.83114 ],
    [2.15911 ],
    [1.20101 ],
    [1.78127 ],
    [1.19561 ]
  ])


  describe "#fit" do
    it "computes discriminant components correctly using tensors as input" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1) # Assuming we want to project down to 1 dimension
      lda.fit(data_tensor, labels)

      lda.components.shape.should eq([1, 2]) # Since we project down to 1 dimension, and have 2 features

      # Check components are similar to expected components
      (lda.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "computes discriminant components correctly using dataframe as input" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1) # Assuming we want to project down to 1 dimension
      lda.fit(data_df, labels)

      lda.components.shape.should eq([1, 2]) # Since we project down to 1 dimension, and have 2 features

      # Check components are similar to expected components
      (lda.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "computes discriminant components correctly using array as input" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1) # Assuming we want to project down to 1 dimension
      lda.fit(data_array, labels)

      lda.components.shape.should eq([1, 2]) # Since we project down to 1 dimension, and have 2 features

      # Check components are similar to expected components
      (lda.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end
  end

  describe "#transform" do
    it "transforms data correctly using tensors" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1)
      lda.fit(data_tensor, labels)

      transformed_data = lda.transform(data_tensor)

      transformed_data.shape.should eq([10, 1]) # 10 samples and 1 dimension as a result of the transformation

      (transformed_data - expected_transformed_data).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "transforms data correctly using dataframes" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1)
      lda.fit(data_df, labels)

      transformed_data = lda.transform(data_df)

      transformed_data.shape.should eq([10, 1]) # 10 samples and 1 dimension as a result of the transformation

      (transformed_data - expected_transformed_data).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "transforms data correctly using arrays" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1)
      lda.fit(data_array, labels)

      transformed_data = lda.transform(data_array)

      transformed_data.shape.should eq([10, 1]) # 10 samples and 1 dimension as a result of the transformation

      (transformed_data - expected_transformed_data).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end
  end
end
