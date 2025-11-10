pub mod data;
pub mod model;
pub mod training;

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn_ndarray::NdArray;
    use crate::model::Model;

    #[test]
    fn test_model_creation() {
        type Backend = NdArray;
        let device = Default::default();
        let _model = Model::<Backend>::new(&device);
    }

    #[test]
    fn test_tensor_operations() {
        type Backend = NdArray;
        let device = Default::default();
        
        let tensor_1 = Tensor::<Backend, 2>::from_floats(
            [[1.0, 2.0], [3.0, 4.0]], 
            &device
        );
        let tensor_2 = Tensor::<Backend, 2>::from_floats(
            [[5.0, 6.0], [7.0, 8.0]], 
            &device
        );
        
        let result = tensor_1 + tensor_2;
        let expected = Tensor::<Backend, 2>::from_floats(
            [[6.0, 8.0], [10.0, 12.0]], 
            &device
        );
        
        let diff = (result - expected).abs();
        let max_diff = diff.max().into_scalar();
        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_model_forward() {
        type Backend = NdArray;
        let device = Default::default();
        let model = Model::<Backend>::new(&device);
        
        let images = Tensor::<Backend, 3>::zeros([2, 28, 28], &device);
        let output = model.forward(images);
        
        assert_eq!(output.dims(), [2, 10]);
    }
}