use ndarray::{ArrayD, Array2, Array1, arr1, arr2,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;

// Enable conversion from f64 to Array1<f64>
// impl From<f64> for Array1<f64> {
//     fn from(x: f64) -> Self {
//         arr1(&[x])
//     }
// }

