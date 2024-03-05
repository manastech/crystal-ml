require "../spec_helper"
require "../../src/dimensionality_reduction/lda"

describe CrystalML::DimensionalityReduction::LDA do
  data_array = [
    [2.5, 2.4], # Class 0
    [0.5, 0.7], # Class 1
    [2.2, 2.9], # Class 0
    [1.9, 2.2], # Class 1
    [3.1, 3.0], # Class 0
    [2.3, 2.7], # Class 1
    [2.0, 1.6], # Class 0
    [1.0, 1.1], # Class 1
    [1.5, 1.6], # Class 0
    [1.1, 0.9]  # Class 1
  ]

  labels = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)

  describe "#fit" do
    it "computes discriminant components correctly using tensors as input" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1) # Assuming we want to project down to 1 dimension
      lda.fit(data_tensor, labels)

      lda.components.shape.should eq([1, 2]) # Since we project down to 1 dimension, and have 2 features

      # LDA does not have explained variance like PCA, but you might want to check for the rank or other properties of the components
      # This part is more about ensuring that the method runs and produces an output of the correct shape and basic properties
    end
  end

  describe "#transform" do
    it "transforms data correctly using tensors" do
      lda = CrystalML::DimensionalityReduction::LDA.new(1)
      lda.fit(data_tensor, labels)

      transformed_data = lda.transform(data_tensor)

      # We won't know the expected transformed data without running the actual LDA algorithm
      # So, this part is more about ensuring that your data is transformed correctly, i.e., it has the right shape
      transformed_data.shape.should eq([10, 1]) # 10 samples and 1 dimension as a result of the transformation


      puts transformed_data
      
      # Here you could also check the variance between the classes in the transformed space if needed,
      # to ensure that LDA is maximizing the class separability
    end
  end
end
