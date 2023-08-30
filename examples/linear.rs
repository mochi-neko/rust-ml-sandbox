// Official hello world: https://huggingface.github.io/candle/guide/hello_world.html
use candle_core::{DType, Device, Result, Tensor};

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?.matmul(&self.weight.contiguous()?)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    // Use Device::Cpu; to use the CPU.
    let device = Device::cuda_if_available(0)?;

    // Creating a dummy model
    let weight = Tensor::zeros((784, 100), DType::F32, &device)?;
    let bias = Tensor::zeros((100,), DType::F32, &device)?;
    let first = Linear { weight, bias };
    let weight = Tensor::zeros((100, 10), DType::F32, &device)?;
    let bias = Tensor::zeros((10,), DType::F32, &device)?;
    let second = Linear { weight, bias };
    let model = Model { first, second };

    let dummy_image = Tensor::zeros((1, 784), DType::F32, &device)?;

    // Inference on the model
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
