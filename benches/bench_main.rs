use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::extract_minimal_with_time::benches,
    benchmarks::extract_minimal::benches,
    benchmarks::extract_comprehensive_with_time::benches,
    benchmarks::extract_comprehensive::benches,
}
