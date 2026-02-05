use ndarray::{ArrayD, Array2, Array1, arr1, arr2,Axis};
use num_complex::Complex64;
use std::f64;
use thiserror::Error;

pub type RhsFunction = Arc<dyn for<'a> Fn(f64, &'a Array1<f64>) -> Array1<f64>>;



pub struct Solver<> {

        f: RhsFunction,
        t: f64,
        t_old: f64,
        t_max: f64,
        dt: f64,
        y: Array1<f64>,
        rtol: f64,
        atol: f64,
        results: SolverResult<Array1<f64>, f64>,
        h: f64,
        h_old: f64,
        n_max_steps: usize,
        stats: SolverStats
}

impl Solver {
    pub fn new(
        f: RhsFunction,
        t: f64,
        t_old: f64,
        t_max: f64,
        dt: f64,
        y: Array1<f64>,
        rtol: f64,
        atol: f64,
        results: SolverResult<Array1<f64>, f64>,
        h: f64,
        h_old: f64,
        n_max_steps: usize,
        stats: SolverStats,
    ) -> Self {
        Solver {
            t: t,
            t_old: t_old,
            tmax: t_max,
            y: y,
            nfev: 0,
            njev: 0,
            nlu: 0,
            status: 0,
            f,
            results: SolverResult::default(),
            stats: SolverStats::new(),
        }
    }

    pub fn solve(&mut self) -> SolverResult<Array1<f64>, f64> {
        let mut y_next=self.y.clone();
        let mut h_opt;
        let mut err;
        while self.t < self.t_max {
            for i in 0..self.n_max_steps {
                (y_next, err) = solver.step(
                    y_next.clone(),
                    self.t,
                    self.h,
                    self.f.clone(),
                    self.rtol,
                );
                
                if err > self.atol  {
                    h_opt = solver.optimal_step_size(self.h, err);
                    self.h = h_opt;
                    break; 
                }
                else {
                    self.h = h_opt;
                }
            }
            self.t += self.h;
            self.y = y_next.clone();
            self.results.push(self.y.clone(), self.t);
            print!("t: {}, y: {}, h: {}\n", self.t, self.y, self.h);
        }
        self.results.clone()
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