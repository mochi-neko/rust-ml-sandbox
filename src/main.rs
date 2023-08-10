mod logging;

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

#[tracing::instrument(name = "main", err)]
fn main() -> anyhow::Result<()> {
    crate::logging::initialize_logging().map_err(|error| {
        println!("Failed to initialize logging: {:?}", error);
        error
    })?;

    tracing::info!("Start main process...");

    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    // Define input
    let sentences = ["this is an example sentence", "each sentence is converted"];

    // Generate Embeddings
    let embeddings = model.encode(&sentences)?;
    println!("{embeddings:?}");

    Ok(())
}
