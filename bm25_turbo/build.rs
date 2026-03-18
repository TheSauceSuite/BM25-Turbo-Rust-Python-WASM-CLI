//! Build script for bm25_turbo.
//!
//! When the `distributed` feature is enabled, generates gRPC service stubs
//! using tonic-build's manual (no-protoc) code generation.

fn main() {
    #[cfg(feature = "distributed")]
    {
        let bm25_shard_service = tonic_build::manual::Service::builder()
            .name("Bm25Shard")
            .package("bm25")
            .comment("BM25 Shard service -- each shard hosts a partition of the corpus.")
            .method(
                tonic_build::manual::Method::builder()
                    .name("search")
                    .route_name("Search")
                    .input_type("crate::distributed::SearchRequest")
                    .output_type("crate::distributed::SearchResponse")
                    .codec_path("tonic_prost::ProstCodec")
                    .build(),
            )
            .method(
                tonic_build::manual::Method::builder()
                    .name("collect_idf")
                    .route_name("CollectIdf")
                    .input_type("crate::distributed::IdfRequest")
                    .output_type("crate::distributed::IdfResponse")
                    .codec_path("tonic_prost::ProstCodec")
                    .build(),
            )
            .method(
                tonic_build::manual::Method::builder()
                    .name("health")
                    .route_name("Health")
                    .input_type("crate::distributed::HealthRequest")
                    .output_type("crate::distributed::HealthResponse")
                    .codec_path("tonic_prost::ProstCodec")
                    .build(),
            )
            .method(
                tonic_build::manual::Method::builder()
                    .name("apply_global_idf")
                    .route_name("ApplyGlobalIdf")
                    .input_type("crate::distributed::GlobalIdfUpdate")
                    .output_type("crate::distributed::GlobalIdfAck")
                    .codec_path("tonic_prost::ProstCodec")
                    .build(),
            )
            .build();

        tonic_build::manual::Builder::new().compile(&[bm25_shard_service]);
    }
}
