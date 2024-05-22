# spec/classification/decision_classifier_spec.cr

require "../spec_helper"
require "../../src/classification/decision_tree_classifier"

describe CrystalML::Classification::DecisionTreeClassifier do
  data_array = [
    [5.1, 7.5, 1.4, 0.2],
    [4.9, 7.0, 1.4, 0.2],
    [3.8, 7.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [5.9, 3.0, 5.1, 1.8]
  ]
  target_array = [0.0, 0.0, 1.0, 1.0, 1.0]

  data_tensor = Tensor(Float64, CPU(Float64)).from_array(data_array)
  target_tensor = Tensor(Float64, CPU(Float64)).from_array(target_array)

  data_df = Crysda.dataframe_of("feature1", "feature2", "feature3", "feature4").values(
    5.1, 7.5, 1.4, 0.2,
    4.9, 7.0, 1.4, 0.2,
    3.8, 7.2, 4.7, 1.4,
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
    [2.1, 9.5, 1.4, 0.2], # Expected class 0
    [2.4, 9.2, 4.5, 1.5]  # Expected class 1
  ]
  test_data_tensor = Tensor(Float64, CPU(Float64)).from_array(test_data_array)
  test_data_dataframe = Crysda.dataframe_of("feature1", "feature2", "feature3", "feature4").values(
    2.1, 9.5, 1.4, 0.2,
    2.4, 9.2, 4.5, 1.5
  )

  describe "#fit" do
    it "correctly builds the decision tree from training data using tensors" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_tensor, target_tensor)

      classifier.root.should_not be_nil
    end

    it "correctly builds the decision tree from training data using arrays" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_array, target_array)

      classifier.root.should_not be_nil
    end

    it "correctly builds the decision tree from training data using dataframes" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_df, target_df)

      classifier.root.should_not be_nil
    end
  end

  describe "#predict" do
    it "correctly predicts the class of given data points using tensors" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_tensor, target_tensor)

      predictions = classifier.predict(test_data_tensor).to_a
      predictions.should eq([0.0, 1.0])
    end

    it "correctly predicts the class of given data points using arrays" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_array, target_array)

      predictions = classifier.predict(test_data_array).to_a
      predictions.should eq([0.0, 1.0])
    end

    it "correctly predicts the class of given data points using dataframes" do
      classifier = CrystalML::Classification::DecisionTreeClassifier.new(max_depth: 3, min_size: 1)
      classifier.fit(data_df, target_df)

      predictions = classifier.predict(test_data_dataframe).to_a
      predictions.should eq([0.0, 1.0])
    end
  end
end
