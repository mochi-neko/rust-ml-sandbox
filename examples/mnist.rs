// https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs
// This should reach 91.5% accuracy.
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, ops, Conv2d, Linear, VarBuilder, VarMap};
use rand::prelude::*;

const LABELS: usize = 10;

trait Model: Sized {
    fn new(vs: VarBuilder) -> Result<Self>;
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor>;
}

#[derive(Debug)]
struct ConvolutionalNetwork {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
}

impl Model for ConvolutionalNetwork {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(
            1,
            32,
            5,
            Default::default(),
            vs.pp("c1"),
        )?;
        let conv2 = candle_nn::conv2d(
            32,
            64,
            5,
            Default::default(),
            vs.pp("c2"),
        )?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        xs.reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    load: Option<String>,
    save: Option<String>,
    epochs: usize,
}

fn training_loop(
    dateset: candle_datasets::vision::Dataset,
    args: &TrainingArgs,
) -> anyhow::Result<()> {
    const BSIZE: usize = 64;

    let device = candle_core::Device::cuda_if_available(0)?;

    // Train dataset
    let train_labels = dateset.train_labels;
    let train_images = dateset
        .train_images
        .to_device(&device)?;
    let train_labels = train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;

    // Model
    let mut var_map = VarMap::new();
    let var_builder_args =
        VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = ConvolutionalNetwork::new(var_builder_args.clone())?;

    if let Some(load) = &args.load {
        println!("loading weights from {load}");
        var_map.load(load)?
    }

    // Optimizer
    let adamw_params = candle_nn::ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut optimizer =
        candle_nn::AdamW::new(var_map.all_vars(), adamw_params)?;

    // Test dataset
    let test_images = dateset
        .test_images
        .to_device(&device)?;
    let test_labels = dateset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let batches = train_images.dim(0)? / BSIZE;
    let mut batch_indices = (0..batches).collect::<Vec<usize>>();

    for epoch in 1..args.epochs {
        let mut sum_loss = 0f32;
        batch_indices.shuffle(&mut thread_rng());

        // Train phase
        for batch_index in batch_indices.iter() {
            let train_images =
                train_images.narrow(0, batch_index * BSIZE, BSIZE)?;
            let train_labels =
                train_labels.narrow(0, batch_index * BSIZE, BSIZE)?;
            let logits = model.forward(&train_images)?;
            let log_softmax = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_softmax, &train_labels)?;
            optimizer.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }
        let avg_loss = sum_loss / batches as f32;

        // Test phase
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }

    if let Some(save) = &args.save {
        println!("saving trained weights in {save}");
        var_map.save(save)?
    }

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    // Load the dataset
    let m = candle_datasets::vision::mnist::load()?;
    println!(
        "train-images: {:?}",
        m.train_images.shape()
    );
    println!(
        "train-labels: {:?}",
        m.train_labels.shape()
    );
    println!(
        "test-images: {:?}",
        m.test_images.shape()
    );
    println!(
        "test-labels: {:?}",
        m.test_labels.shape()
    );

    let training_args = TrainingArgs {
        epochs: 10,
        learning_rate: 0.001,
        load: None,
        save: None,
    };
    training_loop(m, &training_args)
}
