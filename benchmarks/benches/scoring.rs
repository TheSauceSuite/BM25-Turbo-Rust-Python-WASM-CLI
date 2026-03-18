//! Divan microbenchmarks for BM25 scoring kernels.

fn main() {
    divan::main();
}

#[divan::bench(args = ["robertson", "lucene", "atire", "bm25l", "bm25plus"])]
fn bench_idf(bencher: divan::Bencher, variant: &str) {
    let method = match variant {
        "robertson" => bm25_turbo::Method::Robertson,
        "lucene" => bm25_turbo::Method::Lucene,
        "atire" => bm25_turbo::Method::Atire,
        "bm25l" => bm25_turbo::Method::Bm25l,
        "bm25plus" => bm25_turbo::Method::Bm25Plus,
        _ => unreachable!(),
    };

    bencher.bench(|| bm25_turbo::scoring::idf(method, 1_000_000, 500));
}

#[divan::bench(args = ["robertson", "lucene", "atire", "bm25l", "bm25plus"])]
fn bench_tfc(bencher: divan::Bencher, variant: &str) {
    let method = match variant {
        "robertson" => bm25_turbo::Method::Robertson,
        "lucene" => bm25_turbo::Method::Lucene,
        "atire" => bm25_turbo::Method::Atire,
        "bm25l" => bm25_turbo::Method::Bm25l,
        "bm25plus" => bm25_turbo::Method::Bm25Plus,
        _ => unreachable!(),
    };

    bencher.bench(|| bm25_turbo::scoring::tfc(method, 5.0, 120.0, 100.0, 1.5, 0.75, 0.5));
}
