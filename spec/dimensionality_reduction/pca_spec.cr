# spec/dimensionality_reduction/pca_spec.cr

require "../spec_helper"
require "../../src/dimensionality_reduction/pca"

describe CrystalML::DimensionalityReduction::PCA do

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

  expected_components_array = [
    [-0.6778734, -0.73517866],
    [-0.73517866, 0.6778734]
  ]

  expected_transformed_data_array = [
    [-0.82797019, -0.17511531],
    [ 1.77758033,  0.14285723],
    [-0.99219749,  0.38437499],
    [-0.27421042,  0.13041721],
    [-1.67580142, -0.20949846],
    [-0.9129491,   0.17528244],
    [ 0.09910944, -0.3498247 ],
    [ 1.14457216,  0.04641726],
    [ 0.43804614,  0.01776463],
    [ 1.22382056, -0.16267529]
  ]

  describe "#fit" do
    it "computes principal components correctly using tensors as input" do

      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_tensor)

      pca.components.shape.should eq([2, 2])

      pca.explained_variance.sum.should be <= 1.0

      expected_components = Tensor(Float64, CPU(Float64)).from_array(expected_components_array)

      (pca.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "computes principal components correctly using arrays as input" do
      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_array)

      pca.components.shape.should eq([2, 2])

      pca.explained_variance.sum.should be <= 1.0

      expected_components = Tensor(Float64, CPU(Float64)).from_array(expected_components_array)

      (pca.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end

    it "computes principal components correctly using dataframes as input" do
      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_df)

      pca.components.shape.should eq([2, 2])

      pca.explained_variance.sum.should be <= 1.0

      expected_components = Tensor(Float64, CPU(Float64)).from_array(expected_components_array)

      (pca.components - expected_components).each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end
  end

  describe "#transform" do
    it "transforms data correctly using tensors" do
      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_tensor)

      transformed_data = pca.transform(data_tensor)

      expected_transformed_data = Tensor(Float64, CPU(Float64)).from_array(expected_transformed_data_array)

      i=0
      transformed_data.each_axis(0) do |row|
        row.size.should eq(2) # Check dimensionality
        row.to_a.zip(expected_transformed_data[i].to_a).each do |transformed_value, expected_value|
          transformed_value.should be_close(expected_value, 0.01)
        end
        i += 1
      end
    end

    it "transforms data correctly using arrays" do
      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_array)

      transformed_data = pca.transform(data_array)

      expected_transformed_data = Tensor(Float64, CPU(Float64)).from_array(expected_transformed_data_array)

      i=0
      transformed_data.each_axis(0) do |row|
        row.size.should eq(2) # Check dimensionality
        row.to_a.zip(expected_transformed_data[i].to_a).each do |transformed_value, expected_value|
          transformed_value.should be_close(expected_value, 0.01)
        end
        i += 1
      end
    end

    it "transforms data correctly using dataframes" do
      pca = CrystalML::DimensionalityReduction::PCA.new(2)
      pca.fit(data_df)

      transformed_data = pca.transform(data_df)

      expected_transformed_data = Tensor(Float64, CPU(Float64)).from_array(expected_transformed_data_array)

      i=0
      transformed_data.each_axis(0) do |row|
        row.size.should eq(2) # Check dimensionality
        row.to_a.zip(expected_transformed_data[i].to_a).each do |transformed_value, expected_value|
          transformed_value.should be_close(expected_value, 0.01)
        end
        i += 1
      end
    end
  end
end
