use ndarray::Array1;
use odeint::dop54::{DOP54};
use odeint::solver::{SolverStats,SolverResult,RhsFunction, Stepper, Solver};
use std::sync::Arc;


fn example_exponential_decay() {
    println!("=== Example 1: Exponential Decay ===\n");
    
    let k = 0.5;

    let fun: Arc<dyn for<'a> Fn(f64, &'a Array1<f64>) -> Array1<f64>>
    = Arc::new(move |_t: f64, y: &Array1<f64>| {
        y.mapv(|yi| -k * yi)
    });
    
    let y0 = Array1::from_vec(vec![100.0]);
    let method = DOP54::new(y0.clone());
    let mut solver = Solver::new(
        fun,
        method,
        0.0, // t_start
        10.0, // t_max
        y0, // initial condition
        1e-6, // rtol
        1e-6, // atol
        0.1, // initial step size
        0.1, // max step size
        1000, // max steps
    );

    println!("Full solve");
    solver.solve();
    println!("Results: {:?}", solver.results);
}

fn main() {
    example_exponential_decay();
}