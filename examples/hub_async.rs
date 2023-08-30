use candle_core::Device;
// Official hello world: https://huggingface.github.io/candle/guide/hello_world.html
use hf_hub::api::tokio::Api;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api = Api::new()?;
    let repo = api.model("bert-base-uncased".to_string());

    let weights = repo
        .get("model.safetensors")
        .await?;

    let _weights = candle_core::safetensors::load(weights, &Device::Cpu)?;

    // Use weights to do inference

    Ok(())
}
