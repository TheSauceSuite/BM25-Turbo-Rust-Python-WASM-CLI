pub mod dataset;
pub mod eval;
pub mod results;

pub use dataset::{Corpus, Dataset, DatasetFiles, Qrels, Queries};
pub use eval::{dcg, ideal_dcg, mean_ndcg, ndcg_at_k};
pub use results::{BenchmarkRun, DatasetResult};
