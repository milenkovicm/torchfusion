use std::sync::Arc;

use datafusion::execution::{
    config::SessionConfig,
    context::{SessionContext, SessionState},
    runtime_env::{RuntimeConfig, RuntimeEnv},
};
use torchfusion::TorchFunctionFactory;

#[tokio::main]
async fn main() {
    let runtime_config = RuntimeConfig::new();
    let runtime_environment = RuntimeEnv::new(runtime_config).unwrap();
    let session_config =
        SessionConfig::new().set_str("datafusion.sql_parser.dialect", "PostgreSQL");
    let state = SessionState::new_with_config_rt(session_config, Arc::new(runtime_environment))
        .with_function_factory(Arc::new(TorchFunctionFactory {}));
    let ctx = SessionContext::new_with_state(state);

    // a helper function we need
    ctx.register_udf(torchfusion::f32_argmax_udf());

    ctx.register_parquet("iris", "data/iris.snappy.parquet", Default::default())
        .await
        .expect("table to be loaded");

    // we define a torch model to use
    let sql = r#"
    CREATE FUNCTION iris(FLOAT)
    RETURNS FLOAT
    LANGUAGE TORCH
    AS 'model/iris.spt'
    "#;

    ctx.sql(sql).await.unwrap().show().await.unwrap();

    let sql = r#"
        select 
        features, 
        f32_argmax(torch.iris([cast(sl as double),cast(sw as double),cast(pl as double),cast(pw as double)])) as infered_label, 
        label as true_label
        from iris 
        limit 50
    "#;
    ctx.sql(sql).await.unwrap().show().await.unwrap();
}
