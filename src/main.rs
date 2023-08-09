mod logging;

fn main() -> anyhow::Result<()> {
    crate::logging::initialize_logging().map_err(|error| {
        tracing::error!("Failed to initialize logging: {:?}", error);
        error
    })?;

    tracing::info!("Start main process...");

    Ok(())
}
