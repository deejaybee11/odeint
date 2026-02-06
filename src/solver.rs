use ndarray::{ArrayD, Array2, Array1, arr1, arr2,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;
use std::sync::Arc;

pub type RhsFunction = Arc<dyn for<'a> Fn(f64, &'a Array1<f64>) -> Array1<f64>>;

pub trait Stepper {
    fn step(& mut self, y: Array1<f64>, t: f64, h: f64, f: RhsFunction, rtol: f64) -> (Array1<f64>, f64);
    fn optimal_step_size(&self, h: f64, h_max: f64, rtol: f64, err: f64) -> f64;
}

pub struct Solver<M: Stepper> {

        f: RhsFunction,
        t: f64,
        t_start: f64,
        method: M,
        t_old: f64,
        t_max: f64,
        y: Array1<f64>,
        rtol: f64,
        atol: f64,
        pub results: SolverResult<Array1<f64>, f64>,
        h: f64,
        h_old: f64,
        h_max: f64,
        n_max_steps: usize,
        stats: SolverStats,
        status: u32,
        nfev: u32,
        njev: u32,
        nlu: u32,
}

impl<M: Stepper> Solver<M> {
    pub fn new(
        f: RhsFunction,
        method: M,
        t_start: f64,
        t_max: f64,
        y: Array1<f64>,
        rtol: f64,
        atol: f64,
        h: f64,
        h_max: f64,
        n_max_steps: usize,
    ) -> Self {
        Self {
            f,
            method,

            t: t_start,
            t_old: t_start,
            t_start,
            t_max,

            y,

            rtol,
            atol,

            h,
            h_old: h,
            h_max,

            n_max_steps,

            stats: SolverStats::new(),
            status: 0,
            results: SolverResult::default(),
            nfev: 0,
            njev: 0,
            nlu: 0,
        }
    }

    pub fn step(&mut self) {
        
        let mut accept_step = false;
        let mut nsteps = 0;
        let mut h_opt= self.h.clone();
        let mut y_next=self.y.clone();
        let mut err;
        while !accept_step {
            if nsteps > self.n_max_steps {
               accept_step = true;
               //println!("Maximum number of steps exceeded");
            }
            (y_next, err) = self.method.step(
                self.y.clone(),
                self.t,
                self.h,
                self.f.clone(),
                self.rtol,
            );
            if err > self.atol {
                h_opt = self.method.optimal_step_size(self.h, self.h_max, self.rtol, err);
                self.h = h_opt;
            }
            else {
                accept_step = true;
            }      
        }
        self.h = h_opt;
        self.t += self.h;
        self.y = y_next.clone();
        //print!("t: {}, y: {}, h: {}\n", self.t, self.y, self.h)
    }

    pub fn solve(&mut self){
        let mut y_next=self.y.clone();
        let mut h_opt;
        let mut err;
        let mut step_success = false;
        while self.t < self.t_max {

            step_success = false;
            while !step_success {
                (y_next, err) = self.method.step(
                    y_next.clone(),
                    self.t,
                    self.h,
                    self.f.clone(),
                    self.rtol,
                );
                
                if err > self.atol  {
                    h_opt = self.method.optimal_step_size(self.h, self.h_max, self.rtol, err);
                    self.h = h_opt;
                }
                else {
                    step_success = true;
                }
            }
            self.t += self.h;
            self.y = y_next.clone();
            self.results.push(self.y.clone(), self.t);
            //print!("t: {}, y: {}, h: {}\n", self.t, self.y, self.h);
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolverResult<T, V>(Vec<T>, Vec<V>);

impl<T, V> SolverResult<T, V> {
    pub fn new(x: Vec<T>, y: Vec<V>) -> Self {
        SolverResult(x, y)
    }

    pub fn with_capacity(n: usize) -> Self {
        SolverResult(Vec::with_capacity(n), Vec::with_capacity(n))
    }

    pub fn push(&mut self, x: T, y: V) {
        self.0.push(x);
        self.1.push(y);
    }

    pub fn append(&mut self, mut other: SolverResult<T, V>) {
        self.0.append(&mut other.0);
        self.1.append(&mut other.1);
    }

    /// Returns a pair that contains references to the internal vectors
    pub fn get(&self) -> (&Vec<T>, &Vec<V>) {
        (&self.0, &self.1)
    }
}

impl<T, V> Default for SolverResult<T, V> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

#[derive(Debug, Clone)]
pub struct SolverStats {
    pub num_eval: u32,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
}

impl SolverStats {
    pub(crate) fn new() -> SolverStats {
        SolverStats {
            num_eval: 0,
            accepted_steps: 0,
            rejected_steps: 0,
        }
    }
}