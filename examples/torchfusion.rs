#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = torchfusion::configure_context();

    ctx.register_parquet("iris", "data/iris.snappy.parquet", Default::default())
        .await?;

    // ctx.sql("SET torch.cuda_device = 0").await?;
    ctx.sql("SET torch.device = cpu").await?;

    // we define a torch model to use
    let sql = r#"
    CREATE FUNCTION iris(FLOAT[])
    RETURNS FLOAT[]
    LANGUAGE TORCH
    AS 'model/iris.spt'
    "#;

    ctx.sql(sql).await?.show().await?;

    let sql = r#"
        select 
        features, 
        argmax(torch.iris([cast(sl as double),cast(sw as double),cast(pl as double),cast(pw as double)])) as inferred_label, 
        label as true_label
        from iris 
        limit 50
    "#;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
