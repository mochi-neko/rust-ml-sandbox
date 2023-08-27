use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;

fn main() -> anyhow::Result<()> {
    let model = SequenceClassificationModel::new(Default::default())?;

    let input = [
        "This is a positive statement",
        "This is a negative statement",
        "This is a neutral statement",
    ];

    let output = model.predict(input);

    for label in output {
        println!("Label: {:?}", label);
    }

    Ok(())
}
