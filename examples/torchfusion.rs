#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = torchfusion::configure_context();

    let sql = r#"
        CREATE EXTERNAL TABLE iris STORED AS PARQUET LOCATION 'data/iris.snappy.parquet';
    "#;

    ctx.sql(sql).await?.show().await?;

    // ctx.sql("SET torchfusion.cuda_device = 0").await?;
    ctx.sql("SET torchfusion.device = cpu")
        .await?
        .show()
        .await?;

    // definition of torch model to use
    let sql = r#"
        CREATE FUNCTION iris(FLOAT[])
        RETURNS FLOAT[]
        LANGUAGE TORCH
        AS 'model/iris.spt'
    "#;

    ctx.sql(sql).await?.show().await?;

    let sql = r#"
        SELECT 
        sl, sw, pl, pw,
        features, 
        argmax(iris(features)) as f_inferred, 
        argmax(iris([sl, sw, pl, pw])) as inferred, 
        label
        FROM iris 
        LIMIT 50
    "#;

    ctx.sql(sql).await?.show().await?;

    Ok(())
}
