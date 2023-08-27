use rust_bert::pipelines::{
    common::ModelType,
    text_generation::{TextGenerationConfig, TextGenerationModel},
};

fn main() -> anyhow::Result<()> {
    let config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: Some(30),
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };

    let model = TextGenerationModel::new(config)?;

    let context = ["What is the capital of Japan?"];

    let output = model.generate(&context, None);

    for text in output {
        println!("{:?}", text);
    }

    Ok(())
}
