use rust_bert::{
    bart::{BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources},
    pipelines::{
        common::ModelResource,
        summarization::{SummarizationConfig, SummarizationModel},
    },
    resources::RemoteResource,
};
use tch::Device;

fn main() -> anyhow::Result<()> {
    let config_resource = Box::new(RemoteResource::from_pretrained(
        BartConfigResources::DISTILBART_CNN_6_6,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        BartVocabResources::DISTILBART_CNN_6_6,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        BartMergesResources::DISTILBART_CNN_6_6,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        BartModelResources::DISTILBART_CNN_6_6,
    ));

    let config = SummarizationConfig {
        model_resource: ModelResource::Torch(model_resource),
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        num_beams: 1,
        length_penalty: 1.0,
        min_length: 56,
        max_length: Some(142),
        device: Device::Cpu,
        ..Default::default()
    };

    let model = SummarizationModel::new(config)?;

    let input = [
        "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. \
         It's the most aggressive action on tackling the climate crisis in American history, \
         which will lift up American workers and create good-paying, union jobs across the country. \
         It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. \
         And no one making under $400,000 per year will pay a penny more in taxes."
    ];

    let output = model.summarize(&input);

    for summary in output {
        println!("Summary: {:?}", summary);
    }

    Ok(())
}
