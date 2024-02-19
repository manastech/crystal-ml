# spec/classification/naive_bayes_spec.cr

require "../spec_helper"
require "../../src/classification/naive_bayes"

describe CrystalML::Classification::NaiveBayesClassifier do

  data_array = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [5.9, 3.0, 5.1, 1.8]
  ]
  target_array = [0.0, 0.0, 1.0, 1.0, 1.0]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  data_df = Crysda.dataframe_of("feature1", "feature2", "feature3", "feature4").values(
    5.1, 3.5, 1.4, 0.2,
    4.9, 3.0, 1.4, 0.2,
    7.0, 3.2, 4.7, 1.4,
    6.4, 3.2, 4.5, 1.5,
    5.9, 3.0, 5.1, 1.8
  )
  target_df = Crysda.dataframe_of("target").values(
    0.0,
    0.0,
    1.0,
    1.0,
    1.0
  )

  test_data_array = [
    [5.1, 3.5, 1.4, 0.2], # Expected to be classified as 0
    [7.0, 3.2, 4.7, 1.4]  # Expected to be classified as 1
  ]
  test_data_tensor = Tensor(Float64, CPU(Float64)).from_array(test_data_array)
  test_data_dataframe = Crysda.dataframe_of("feature1", "feature2", "feature3", "feature4").values(
    5.1, 3.5, 1.4, 0.2,
    7.0, 3.2, 4.7, 1.4
  )
  describe "#fit" do
    it "correctly computes means and variances for each class using tensors" do
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_tensor, target_tensor)

      # Expected means and variances for each class
      expected_means_for_class_0 = [5.0, 3.25, 1.4, 0.2]
      expected_variances_for_class_0 = [0.01, 0.0625, 0.0, 0.0]
      expected_means_for_class_1 = [6.4333, 3.1333, 4.7666, 1.5666]
      expected_variances_for_class_1 = [0.2022, 0.0088, 0.0622, 0.0288]

      expected_means_for_class_0.each_with_index do |val, i|
        classifier.means[0.0][i].value.should be_close(val, 0.01)
      end
    
      expected_variances_for_class_0.each_with_index do |val, i|
        classifier.variances[0.0][i].value.should be_close(val, 0.01)
      end
      expected_means_for_class_1.each_with_index do |val, i|
        classifier.means[1.0][i].value.should be_close(val, 0.01)
      end
      expected_variances_for_class_1.each_with_index do |val, i|
        classifier.variances[1.0][i].value.should be_close(val, 0.01)
      end
    end

    it "correctly computes means and variances for each class using arrays" do
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_array, target_array)

      # Expected means and variances for each class 
      expected_means_for_class_0 = [5.0, 3.25, 1.4, 0.2]
      expected_variances_for_class_0 = [0.01, 0.0625, 0.0, 0.0]
      expected_means_for_class_1 = [6.4333, 3.1333, 4.7666, 1.5666]
      expected_variances_for_class_1 = [0.2022, 0.0088, 0.0622, 0.0288]

      expected_means_for_class_0.each_with_index do |val, i|
        classifier.means[0.0][i].value.should be_close(val, 0.01)
      end
    
      expected_variances_for_class_0.each_with_index do |val, i|
        classifier.variances[0.0][i].value.should be_close(val, 0.01)
      end
      expected_means_for_class_1.each_with_index do |val, i|
        classifier.means[1.0][i].value.should be_close(val, 0.01)
      end
      expected_variances_for_class_1.each_with_index do |val, i|
        classifier.variances[1.0][i].value.should be_close(val, 0.01)
      end
    end

    it "correctly computes means and variances for each class using dataframes" do
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_df, target_df)

      # Expected means and variances for each class 
      expected_means_for_class_0 = [5.0, 3.25, 1.4, 0.2]
      expected_variances_for_class_0 = [0.01, 0.0625, 0.0, 0.0]
      expected_means_for_class_1 = [6.4333, 3.1333, 4.7666, 1.5666]
      expected_variances_for_class_1 = [0.2022, 0.0088, 0.0622, 0.0288]

      expected_means_for_class_0.each_with_index do |val, i|
        classifier.means[0.0][i].value.should be_close(val, 0.01)
      end
    
      expected_variances_for_class_0.each_with_index do |val, i|
        classifier.variances[0.0][i].value.should be_close(val, 0.01)
      end
      expected_means_for_class_1.each_with_index do |val, i|
        classifier.means[1.0][i].value.should be_close(val, 0.01)
      end
      expected_variances_for_class_1.each_with_index do |val, i|
        classifier.variances[1.0][i].value.should be_close(val, 0.01)
      end
    end

  end

  describe "#predict" do
    it "correctly predicts the class of a given data point using tensors" do
      
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_tensor, target_tensor)

      predictions = classifier.predict(test_data_tensor).to_a
      predictions.should eq([0.0, 1.0])
    end

    it "correctly predicts the class of a given data point using arrays" do
      
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_array, target_array)

      predictions = classifier.predict(test_data_array).to_a
      predictions.should eq([0.0, 1.0])
    end

    it "correctly predicts the class of a given data point using dataframes" do
      
      classifier = CrystalML::Classification::NaiveBayesClassifier.new
      classifier.fit(data_df, target_df)

      predictions = classifier.predict(test_data_dataframe).to_a
      predictions.should eq([0.0, 1.0])
    end

  end
end
