# spec/clustering/k_means_spec.cr

require "../spec_helper"
require "../../src/clustering/k_means"

describe CrystalML::Clustering::KMeans do
  data_array = [
    [1.0, 2.0],
    [1.5, 1.8],
    [23.0, 18.0],
    [23.0, 18.5],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
  ]
  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  data_df = Crysda.dataframe_of("feature1", "feature2").values(
    1.0, 2.0,
    1.5, 1.8,
    23.0, 18.0,
    23.0, 18.5,
    1.0, 0.6,
    9.0, 11.0,
    8.0, 2.0,
    10.0, 2.0,
    9.0, 3.0
  )

  describe "#fit and #predict" do
    it "clusters data correctly using tensors" do

      # Create KMeans instance
      kmeans = CrystalML::Clustering::KMeans.new(n_clusters: 3)
      kmeans.fit(data_tensor)

      # Predict clusters
      predictions = kmeans.predict(data_tensor)

      # Check if the number of unique clusters matches the number of centroids
      unique_clusters = predictions.uniq.size
      unique_clusters.should eq 3
    end

    it "clusters data correctly using dataframes" do
      kmeans = CrystalML::Clustering::KMeans.new(n_clusters: 3)
      kmeans.fit(data_df)

      predictions = kmeans.predict(data_df)

      unique_clusters = predictions.uniq.size
      unique_clusters.should eq 3
    end

    it "clusters data correctly using arrays" do
      kmeans = CrystalML::Clustering::KMeans.new(n_clusters: 3)
      kmeans.fit(data_array)

      predictions = kmeans.predict(data_array)

      unique_clusters = predictions.uniq.size
      unique_clusters.should eq 3
    end
  end
end
