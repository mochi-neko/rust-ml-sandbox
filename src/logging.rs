use anyhow::Result;
use tracing::{Event, Level, Subscriber};
use tracing_appender::non_blocking;
use tracing_subscriber::{
    filter::EnvFilter,
    fmt::{
        self,
        format::{self, FmtSpan},
        FmtContext, FormatEvent, FormatFields,
    },
    prelude::__tracing_subscriber_SubscriberExt,
    registry::LookupSpan,
    Registry,
};

pub(crate) fn initialize_logging() -> Result<()> {
    // Set up logging to stdout
    let stdout_layer = fmt::layer()
        .with_target(true)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(true)
        .event_format(StdLogFormatter);

    // Set up logging to a file
    let (non_blocking, _guard) = non_blocking(tracing_appender::rolling::never(
        ".logs",
        format!(
            "rust-ml-sandbox-{}.log",
            chrono::Local::now().format("%Y%m%d-%H%M%S%3f")
        ),
    ));

    let file_layer = fmt::layer()
        .with_target(true)
        .with_writer(non_blocking)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(false)
        .event_format(FileLogFormatter);

    let subscriber = Registry::default()
        .with(stdout_layer)
        .with(file_layer)
        .with(EnvFilter::from_default_env());

    tracing::subscriber::set_global_default(subscriber).map_err(|error| {
        tracing::error!("Failed to set global default subscriber: {:?}", error);
        error
    })?;

    Ok(())
}

pub(crate) struct StdLogFormatter;

impl<S, N> FormatEvent<S, N> for StdLogFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let metadata = event.metadata();

        // Write metadata with color
        write!(
            writer,
            "{:<5} {} {} in {}",
            color_for_log_level(*metadata.level()).paint(metadata.level().to_string()),
            color_for_weaked().paint(
                chrono::Local::now()
                    .format("%Y-%m-%d %H:%M:%S%.3f%Z")
                    .to_string()
            ),
            color_for_weaked().paint(thread_id::get().to_string()),
            color_for_weaked().paint(format!(
                "{}({})",
                metadata.target(),
                metadata.line().unwrap_or_default()
            )),
        )?;

        // Write spans
        format_spans(&mut writer, ctx)?;

        writeln!(writer)?;

        // Write the event
        ctx.format_fields(writer.by_ref(), event)?;

        // Write a new line
        writeln!(writer)?;
        writeln!(writer)
    }
}

fn color_for_log_level(level: Level) -> ansi_term::Color {
    match level {
        Level::TRACE => ansi_term::Color::Fixed(13), // Fuchsia
        Level::DEBUG => ansi_term::Color::Fixed(14), // Aqua
        Level::INFO => ansi_term::Color::Fixed(10),  // Lime
        Level::WARN => ansi_term::Color::Yellow,
        Level::ERROR => ansi_term::Color::Red,
    }
}

fn color_for_weaked() -> ansi_term::Color {
    ansi_term::Color::Fixed(8) // Grey
}

fn format_spans<S, N>(
    writer: &mut impl std::fmt::Write,
    ctx: &FmtContext<'_, S, N>,
) -> std::fmt::Result
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    let mut counter = 0;

    ctx.visit_spans(|span| {
        counter += 1;

        if counter == 1 {
            write!(writer, "... ")?;
        } else {
            write!(writer, " -> ")?;
        }

        // Write span metadata with color
        write!(writer, "{}", span.name())
    })?;

    Ok(())
}

pub(crate) struct FileLogFormatter;

impl<S, N> FormatEvent<S, N> for FileLogFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let metadata = event.metadata();

        // Write metadata with color
        writeln!(
            writer,
            "[{:<5}] {} {} {}({}) ",
            metadata.level(),
            thread_id::get(),
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f%Z"),
            metadata.target(),
            metadata.line().unwrap_or_default(),
        )?;

        // Write spans
        format_spans(&mut writer, ctx)?;

        write!(writer, " ")?;

        // Write the event
        ctx.format_fields(writer.by_ref(), event)?;

        // Write line
        writeln!(writer)
    }
}
