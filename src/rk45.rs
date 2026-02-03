use ndarray::{ArrayD, Array2, Array1, arr1, arr2,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;

pub type RhsFunction = Box<dyn Fn(f64, &Array1<f64>) -> Array1<f64>>;

pub fn rk45_step(
   y: &Array1<f64>,
   t: f64, 
   h: f64, 
   f: RhsFunction,
   tolerance: f64,
) -> Array1<f64> {
    
    let mut k1 = h * f(t,y);
    let mut k2 = h * f(t + 1./5. * h, y + 1./5. * &k1);
    let mut k3 = h * f(t + 3./10. * h, y + 3./40. * &k1 + 9./40 * &k2);
    let mut k4 = h * f(t + 4./5. * h, y + 44./45. * &k1 - 56./15. * &k2 + 32./9. * &k3);
    let mut k5 = h * f(t + 8./9. * h, y + 19372./6561. * &k1 - 25360./2187. * &k2 + 64448./6561. * &k3 - 212./729. * &k4);
    let mut k6 = h * f(t + h, y + 9017./3168. * &k1 - 355./33. * &k2 + 46732./5247. * &k3 + 49./176. * &k4 - 5103./18656. * &k5);
    let mut k7 = h * f(t + h, y + 35./384. * &k1 + 500./1113. * &k3 + 125./192. * &k4 - 2187./6784. * &k5 + 11./84. * &k6);
    let y_next = y + 35./384. * &k1 + 500./1113. * &k3 + 125./192. * &k4 - 2187./6784. * &k5 + 11./84. * &k6;

    let zk1 = y + 517./576. * &k1 + 7571./16695. * &k3 + 393./640. * &k4 - 92097./339200. * &k5 + 187./2100. * &k6 + 1./40. * &k7;
    let error_estimate = y_next - zk1;
    let hopt = 0.9 * h * (tolerance / (2. * zk1 - y_next)).powf(0.2);
    
    y_next
}