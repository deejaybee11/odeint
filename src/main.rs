use ndarray::Array1;
use runge_kutta::solver::{RK45,solve_ivp,OdeSolver};

fn example_exponential_decay() {
    println!("=== Example 1: Exponential Decay ===\n");
    
    let k = 0.5;
    let fun = Box::new(move |_t: f64, y: &Array1<f64>| {
        y.mapv(|yi| -k * yi)
    });
    
    let y0 = Array1::from_vec(vec![100.0]);

    let (y_final, h_opt) = rk45_step(&y0, 0.0, 0.1, fun, 1e-6);
    println!("Final value: {:?}", y_final);
    println!("Optimal step size: {}", h_opt);
}

fn main() {
    example_exponential_decay();
}