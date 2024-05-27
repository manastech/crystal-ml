# Crystal-ML
A classic machine learning library for Crystal programming language, inspired by scikit-learn. 

Crystal-ML focuses on simplicity and ease of use, accepting `Array`, `Tensor` (Num.cr) and/or `DataFrame`(Crystal-DA) objects as inputs in all its algorithms. All the calculations rely on `Tensor` operations, enabling efficient computation and ─future─ support for GPU operations.

## Installation
Add this to your application's `shard.yml`:

```yaml
dependencies:
  crystal-ml:
    github: manastech/crystal-ml
```

Then, run `shards install`.

## Usage

Let's take as an example the `KMeans` algorithm, that partitions data into K distinct clusters based on distance to the centroid of each cluster.

Example:

```
require "crystal-ml"

# Sample data: array of 2D points
data = [
  [1.0, 2.0],
  [1.5, 1.8],
  [5.0, 8.0],
  [8.0, 8.0],
  [1.0, 0.6],
  [9.0, 11.0],
]

# Create a KMeans instance with 3 clusters
kmeans = CrystalML::Clustering::KMeans.new(n_clusters: 3)

# Fit the model to your data
kmeans.fit(data)

# Predict the closest cluster for each data point
predictions = kmeans.predict(data)

puts "Cluster assignments: #{predictions}"
```

The `kmeans` instance (and the rest of the algorithms also) will work seamlessly if the input is a `Tensor`:

```
# require "num" 

data_tensor = Tensor(Float64, CPU(Float64)).from_array(data)
```

Or a `DataFrame`:

```
# require "crysda"

data_df = Crysda.dataframe_of("feature1", "feature2").values(
  1.0, 2.0,
  1.5, 1.8,
  5.0, 8.0,
  8.0, 8.0,
  1.0, 0.6,
  9.0, 11.0
)
```

The return values for `predict`, `fit` and `transform` methods along the library will always have `Tensor` type, leaving to the user its proper conversion if needed. 

## Features & development plan

#### Algorithms

- Clustering
  - [x] KMeans
  - [ ] Affinity propagation
  - [ ] Spectral
  - [ ] DBSCAN
  - [ ] ...
- Classification
  - [x] NaiveBayes
  - [x] Ridge (Binary)
  - [x] Decision trees
  - [x] Random forest
  - [ ] Gradient boosting
  - [ ] Nearest neighbors
  - [ ] ...
- Regression
  - [x] Linear
  - [x] Bayesian Ridge
  - [x] Decision trees
  - [x] Random forest
  - [ ] Ordinary Least Squares
  - [ ] Nearest neighbors
  - [ ] ...
- Transformation
  - [x] PCA
  - [x] LDA
  - [ ] ICA
  - [ ] kPCA
  - [ ] ...

#### Other features

- [ ] Ensembles
- [ ] GPU support

## Development
To run all tests:

```
crystal spec
```

## Contributing

- Fork it (https://github.com/manastech/crystal-ml/fork)
- Create your feature branch (`git checkout -b my-new-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin my-new-feature`)
- Create a new Pull Request

## Contributors

- Leandro Radusky - creator and mantainer.