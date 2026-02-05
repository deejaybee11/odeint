use ndarray::Array1;
use odeint::dop54::{DOP54, RhsFunction};
use odeint::solver::{SolverStats,SolverResult};
use std::sync::Arc;


fn example_exponential_decay() {
    println!("=== Example 1: Exponential Decay ===\n");
    
    let k = 0.5;

    let fun: Arc<dyn for<'a> Fn(f64, &'a Array1<f64>) -> Array1<f64>>
    = Arc::new(move |_t: f64, y: &Array1<f64>| {
        y.mapv(|yi| -k * yi)
    });
    
    let y0 = Array1::from_vec(vec![100.0]);

    let mut solver = DOP54::new(
        fun.clone(),
        0.0,
        y0,
        10.0,
        0.1,
        1e-6,
        1e-6,
        20,
    );

    let result = solver.solve();
}

fn main() {
    example_exponential_decay();
}