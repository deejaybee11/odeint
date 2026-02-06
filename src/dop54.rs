use ndarray::{ArrayD, Array2, Array1, arr1, arr2, array,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;
use crate::solver::{SolverResult, SolverStats, RhsFunction, Stepper, Solver};

pub struct DOP54 {

    A: Array2<f64>,
    B: Array1<f64>,
    C: Array1<f64>,
    K: Vec<Array1<f64>>,
    y_next: Array1<f64>,
}

impl DOP54 {
    pub fn new(y: Array1<f64>) -> Self {
        DOP54 {
            A: array![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1./5., 0.0, 0.0, 0.0, 0.0, 0.0],
                    [3./40., 9./40., 0.0, 0.0, 0.0, 0.0],
                    [44./45., -56./15., 32./9., 0.0, 0.0, 0.0],
                    [19372./6561., -25360./2187., 64448./6561., -212./729., 0.0, 0.0],
                    [9017./3168., -355./33., 46732./5247., 49./176., -5103./18656., 0.0]],
            B: array![1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0],
            C: array![5179./57600., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.],
            K: vec![Array1::<f64>::zeros(y.len()); 7],
            y_next: Array1::<f64>::zeros(y.len()),    
        }
    }
}

impl Stepper for DOP54 {
    fn step(&mut self,
        y: Array1<f64>,
        t: f64, 
        h: f64, 
        f: RhsFunction,
        tolerance: f64,
        ) -> (Array1<f64>, f64) {

            let mut yk = y.clone();
            let mut y_next = Array1::<f64>::zeros(y.len());
            self.K[0] = h * f(t , &yk);
            self.K[1] = h * f(t + self.B[0] * h, &(&yk + (self.A[(0,0)] * &self.K[0])));
            self.K[2] = h * f(t + self.B[1] * h, &(&yk + (self.A[(1,0)] * &self.K[0] + self.A[(1,1)] * &self.K[1])));
            self.K[3] = h * f(t + self.B[2] * h, &(&yk + (self.A[(2,0)] * &self.K[0] + self.A[(2,1)] * &self.K[1] + self.A[(2,2)] * &self.K[2])));
            self.K[4] = h * f(t + self.B[3] * h, &(&yk + (self.A[(3,0)] * &self.K[0] + self.A[(3,1)] * &self.K[1] + self.A[(3,2)] * &self.K[2] + self.A[(3,3)] * &self.K[3])));
            self.K[5] = h * f(t + self.B[4] * h, &(&yk + (self.A[(4,0)] * &self.K[0] + self.A[(4,1)] * &self.K[1] + self.A[(4,2)] * &self.K[2] + self.A[(4,3)] * &self.K[3] +self.A[(4,4)] * &self.K[4])));
            self.K[6] = h * f(t +self.B [5]*h,&(&yk+(self.A[(5,0)]*&self.K[0]+self.A[(5,1)]*&self.K[1]+self.A[(5,2)]*&self.K[2] + self.A[(5,3)] * &self.K[3] + self.A[(5,4)] * &self.K[4] + self.A[(5,5)] * &self.K[5])));
            y_next = &yk + self.A[(5,0)] * &self.K[0] + self.A[(5,1)] * &self.K[1] + self.A[(5,2)] * &self.K[2] + self.A[(5,3)] * &self.K[3] + self.A[(5,4)] * &self.K[4] + self.A[(5,5)] * &self.K[5];
            let zk1:Array1<f64> = &yk +(self.C[0] * &self.K[0] + self.C[1] * &self.K[1] + self.C[2] * &self.K[2] + self.C[3] * &self.K[3] + self.C[4] * &self.K[4]+	self.C[5] * &self.K[5]);
            let _error = 2.0*(y_next.clone() - zk1.clone()).abs()[0];
            
            (y_next, _error)
        }

    fn optimal_step_size(&self, h: f64, h_max: f64, rtol: f64, err: f64) -> f64 {
        let safety = 0.9;
        let exponent = 1.0 / 5.0;
        let h_opt = safety * h * (rtol / err).powf(exponent);
        h_opt.min(h_max)
    }
}