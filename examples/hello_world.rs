// Official hello world: https://huggingface.github.io/candle/guide/hello_world.html
use candle_core::{DType, Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn forward(
        &self,
        image: &Tensor,
    ) -> Result<Tensor> {
        let x = image
            .contiguous()?
            .matmul(&self.first.contiguous()?)?;
        let x = x.relu()?;
        x.contiguous()?
            .matmul(&self.second.contiguous()?)
    }
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let first = Tensor::zeros((784, 100), DType::F32, &device)?;
    let second = Tensor::zeros((100, 10), DType::F32, &device)?;

    let model = Model {
        first,
        second,
    };

    let dummy_image = Tensor::zeros((1, 784), DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");

    Ok(())
}
