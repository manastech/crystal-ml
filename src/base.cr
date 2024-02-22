require "num"
require "crysda"

module CrystalML
  abstract class BaseEstimator
    protected def to_tensor(data : Crysda::DataFrame | Array(Array(Float64))| Array(Float64)) : Tensor(Float64, CPU(Float64))
      data_array = data.is_a?(Crysda::DataFrame) ? data.rows.map { |row| row.values.map(&.as_f) } : data
      Tensor(Float64, CPU(Float64)).from_array(data_array.to_a)
    end
  end

  abstract class SupervisedEstimator < BaseEstimator
    abstract def fit(data : Tensor(Float64, CPU(Float64)), target : Tensor(Float64, CPU(Float64)))
    abstract def predict(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))

    def fit(data : Tensor(Float64, CPU(Float64)) | Array(Array(Float64)) | Crysda::DataFrame,
            target : Tensor(Float64, CPU(Float64)) | Array(Float64) | Crysda::DataFrame)
      data_tensor = data.is_a?(Tensor) ? data : to_tensor(data)
      target_tensor = target.is_a?(Tensor) ? target : to_tensor(target)
      fit(data_tensor, target_tensor)
    end

    def predict(data : Array(Array(Float64)) | Crysda::DataFrame) : Tensor(Float64, CPU(Float64))
      data_tensor = data.is_a?(Tensor) ? data : to_tensor(data)
      predict(data_tensor)
    end
  end

  abstract class UnsupervisedEstimator < BaseEstimator
    abstract def fit(data : Tensor(Float64, CPU(Float64)))

    def fit(data : Array(Array(Float64)) | Crysda::DataFrame)
      data_tensor = data.is_a?(Tensor) ? data : to_tensor(data)
      fit(data_tensor)
    end
  end

  abstract class Regressor < SupervisedEstimator
  end

  abstract class Classifier < SupervisedEstimator
  end

  abstract class Clusterer < UnsupervisedEstimator
    abstract def predict(data : Tensor(Float64, CPU(Float64))) : Array(Int32)

    def predict(data : Array(Array(Float64)) | Crysda::DataFrame) : Array(Int32)
      data_tensor = data.is_a?(Tensor) ? data : to_tensor(data)
      predict(data_tensor).to_a
    end
  end

  abstract class Transformer < UnsupervisedEstimator
    abstract def transform(data : Tensor(Float64, CPU(Float64))) : Tensor(Float64, CPU(Float64))

    def transform(data : Array(Array(Float64)) | Crysda::DataFrame) : Tensor(Float64, CPU(Float64))
      data_tensor = data.is_a?(Tensor) ? data : to_tensor(data)
      transform(data_tensor)
    end
  end
end
