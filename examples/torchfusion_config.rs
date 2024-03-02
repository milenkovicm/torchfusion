#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = torchfusion::configure_context();

    ctx.sql("SET torch.cuda_device = 0").await?;
    ctx.sql("SET torch.device = cpu").await?;

    Ok(())
}
