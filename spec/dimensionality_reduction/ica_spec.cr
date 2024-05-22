# spec/dimensionality_reduction/ica_spec.cr

require "../spec_helper"
require "../../src/dimensionality_reduction/ica"

describe CrystalML::DimensionalityReduction::ICA do

  # Example mixed signal data
  mixed_signals_array = [
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

  mixed_signals_tensor = Tensor(Float64, CPU(Float64)).from_array(mixed_signals_array)

  # Expected independent components - Placeholder for expected results
  # Note: The actual expected components will depend on your specific dataset and ICA initialization
  expected_components_array = [
    [0.37330574, -0.64462611],
    [0.11015664, -0.7954656 ]
  ]

  describe "#fit" do
    it "separates mixed signals into independent components using tensors as input" do
      ica = CrystalML::DimensionalityReduction::ICA.new(2)
      ica.fit(mixed_signals_tensor)

      ica.components.shape.should eq([2, 2])

      expected_components = Tensor(Float64, CPU(Float64)).from_array(expected_components_array)

      # Because ICA components can be scaled and/or inverted, we check for absolute values
      ica_components_abs = ica.components.map { |x| x.abs }
      expected_components_abs = expected_components.map { |x| x.abs }

      puts "***************"
      puts ica_components_abs
      puts expected_components_abs
      puts "***************"

      diff = ica_components_abs - expected_components_abs

      diff.each do |value|
        value.abs.should be_close(0.0, 0.01)
      end
    end
  end

#   describe "#transform" do
#     it "transforms mixed signals back to independent components" do
#       ica = CrystalML::DimensionalityReduction::ICA.new(2)
#       ica.fit(mixed_signals_tensor)

#       transformed_data = ica.transform(mixed_signals_tensor)

#       # Placeholder for expected transformed data
#       # Note: This will need to be filled in based on the expected outcome of your ICA on the test data
#       expected_transformed_data_array = [
#         # Expected independent components after transformation
#       ]
#       expected_transformed_data = Tensor(Float64, CPU(Float64)).from_array(expected_transformed_data_array)

#       i=0
#       transformed_data.each_axis(0) do |row|
#         row.size.should eq(2) # Check dimensionality
#         row.to_a.zip(expected_transformed_data[i].to_a).each do |transformed_value, expected_value|
#           transformed_value.should be_close(expected_value, 0.01)
#         end
#         i += 1
#       end
#     end
#   end
end
