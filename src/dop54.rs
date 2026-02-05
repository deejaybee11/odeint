use ndarray::{ArrayD, Array2, Array1, arr1, arr2,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;
use crate::solver::{SolverResult, SolverStats, RhsFunction};
use std::sync::Arc;

pub struct DOP54<> {

    A: Array2<f64>::[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1./5., 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3./40., 9./40., 0.0, 0.0, 0.0, 0.0],
                    [44./45., -56./15., 32./9., 0.0, 0.0, 0.0],
                    [19372./6561., -25360./2187., 64448./6561., -212./729., 0.0, 0.0],
                    [9017./3168., -355./33., 46732./5247., 49./176., -5103./18656., 0.0]],
    B: Array1<f64>::[1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0],
    C: Array1<f64>::[5179./57600., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.],
    K: Vec<Array1<f64>>::vec![Array1::<f64>::zeros(y.len()); 7],
    y_next: Array1::<f64>::zeros(y.len());
}

impl DOP54<> {
    pub fn new() -> Self {
        DOP54 {
        
        }
    }

    pub fn optimal_step_size(&self, h: f64, err: f64) -> f64 {
        let safety = 0.9;
        let exponent = 1.0 / 5.0;
        let h_opt = safety * h * (self.rtol / err).powf(exponent);
        h_opt.min(self.h_max)
    }

    pub fn step(
    y: Array1<f64>,
    t: f64, 
    h: f64, 
    f: RhsFunction,
    tolerance: f64,
    ) -> (Array1<f64>, f64) {
        

        for i in 0..7 {

            y_next += y_next + A[i] * k[i]

        }

        k[0] = h * f(t,&y_next);
        k[1] = h * f(t + 1./5. * h, &(&y_next + 1./5. * &k[0]));
        k[2] = h * f(t + 3./10. * h, &(&y_next + 3./40. * &k[0] + 9./40. * &k[1]));
        k[3] = h * f(t + 4./5. * h, &(&y_next + 44./45. * &k[0] - 56./15. * &k[1] + 32./9. * &k[2]));
        k[4] = h * f(t + 8./9. * h, &(&y_next + 19372./6561. * &k[0] - 25360./2187. * &k[1] + 64448./6561. * &k[2] - 212./729. * &k[3]));
        k[5] = h * f(t + h, &(&y_next + 9017./3168. * &k[0] - 355./33. * &k[1] + 46732./5247. * &k[2] + 49./176. * &k[3] - 5103./18656. * &k[4]));
        k[6] = h * f(t + h, &(&y_next + 35./384. * &k[0] + 500./1113. * &k[2] + 125./192. * &k[3] - 2187./6784. * &k[4] + 11./84. * &k[5]));
        y_next = y_next + 35./384. * &k[0] + 500./1113. * &k[2] + 125./192. * &k[3] - 2187./6784. * &k[4] + 11./84. * &k[5];
        let zk1 = y + 5179./57600. * &k[0] + 7571./16695. * &k[2] + 393./640. * &k[3] - 92097./339200. * &k[4] + 187./2100. * &k[5] + 1./40. * &k[6];
        let _error = 2*(y_next.clone() - zk1.clone()).abs()[0];
        
        (y_next, _error_estimate)
    }

}