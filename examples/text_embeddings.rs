use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let input = ["this is an example sentence", "each sentence is converted"];

    let output = model.encode(&input)?;

    for embedding in output {
        println!("Embedding : {:?}", embedding);
    }

    Ok(())
}
