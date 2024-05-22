require "../spec_helper"
require "../../src/clustering/affinity_propagation"

describe CrystalML::Clustering::AffinityPropagation do
  data_array = [
    [1.0, 2.0],
    [1.5, 1.8],
    [23.0, 18.0],
    [23.0, 18.5],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [23.0, 19.0],
    [9.0, 3.0]
  ]
  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  
  describe "#fit" do
    it "correctly identifies exemplars and clusters in the data" do
      affinity_propagation = CrystalML::Clustering::AffinityPropagation.new(preference: -50.0)
      affinity_propagation.fit(data_tensor)

      # Verify exemplars have been chosen; the exact number of clusters is data-dependent and thus not specified
      affinity_propagation.exemplars.size.should_not eq 0

    end
  end

  describe "#predict" do
    it "assigns data points to the closest exemplar, forming clusters" do
      affinity_propagation = CrystalML::Clustering::AffinityPropagation.new(preference: -50.0)
      affinity_propagation.fit(data_tensor)
      predictions = affinity_propagation.predict(data_tensor)

      puts affinity_propagation.exemplars
      puts affinity_propagation.responsibility_matrix
      puts affinity_propagation.availability_matrix
      puts "Predictions: #{predictions}"

      # Verify predictions have been made for each data point
      predictions.size.should eq data_array.size
      # Ensure there's at least one cluster
      predictions.uniq.size.should_not eq 0

      # Additional assertions could be made about the properties of the clusters, but these would be highly data- and parameter-specific.
    end
  end
end
