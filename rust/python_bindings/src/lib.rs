use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn greet(name: &str) -> PyResult<String> {
    Ok(format!("Hello, {}!", name))
}

#[pymodule]
fn python_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Use add_wrapped instead of add_function:
    m.add_function(wrap_pyfunction!(greet, m)?)?;
    Ok(())
}
