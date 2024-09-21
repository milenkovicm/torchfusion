use datafusion::{
    arrow::datatypes::DataType,
    common::config_err,
    execution::{
        config::SessionConfig,
        context::{FunctionFactory, RegisterFunction, SessionContext, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
        SessionStateBuilder,
    },
    logical_expr::{CreateFunction, ScalarUDF},
    scalar::ScalarValue,
};
use log::debug;
use std::sync::Arc;

mod argmax;
mod config;
mod udf;

pub use argmax::*;
pub use config::*;

#[derive(Default, Debug)]
pub struct TorchFunctionFactory {}

#[async_trait::async_trait]
impl FunctionFactory for TorchFunctionFactory {
    async fn create(
        &self,
        state: &SessionState,
        statement: CreateFunction,
    ) -> datafusion::error::Result<RegisterFunction> {
        let model_name = statement.name;

        let arg_data_type = statement
            .args
            .map(|a| {
                a.first()
                    .map(|r| r.data_type.clone())
                    .unwrap_or(DataType::Float32)
            })
            .unwrap_or(DataType::Float32);

        let data_type_input = find_item_type(&arg_data_type);

        let data_type_return = statement
            .return_type
            .map(|t| find_item_type(&t))
            .unwrap_or(data_type_input.clone());

        let config = state
            .config()
            .options()
            .extensions
            .get::<TorchConfig>()
            .expect("torch configuration extension to be configured");

        let model_file = match statement.params.function_body {
            Some(datafusion::prelude::Expr::Literal(ScalarValue::Utf8(Some(s)))) => s,
            _ => config_err!("model file should be specified")?,
        };
        // not sure if `ListingTableUrl` is the best way forward
        // but it works.
        let model_file_url = datafusion::datasource::listing::ListingTableUrl::parse(model_file)?;

        log::debug!("loading model from url: {}", model_file_url);

        let object_store = state
            .runtime_env()
            .object_store_registry
            .get_store(model_file_url.as_ref())?;

        let model_file = object_store
            .get(model_file_url.prefix())
            .await?
            .bytes()
            .await?;

        let mut model_file = &model_file[..];

        // function will use same device as long as it is
        // not dropped. if change of defice is needed
        // function should be droped and re-created.

        let device = config.device();
        let batch_size = config.batch_size();
        let model_udf = udf::load_torch_model(
            &model_name,
            &mut model_file,
            device,
            batch_size,
            data_type_input,
            data_type_return,
        )?;

        debug!("Registering function: [{:?}]", model_udf);

        Ok(RegisterFunction::Scalar(Arc::new(model_udf)))
    }
}

fn find_item_type(dtype: &DataType) -> DataType {
    match dtype {
        // We're interested in array type not the array.
        // There is discrepancy between array type defined by create function
        // `List(Field { name: \"field\", data_type: Float32, nullable:  ...``
        // and array type defined by create array operation
        //`[List(Field { name: \"item\", data_type: Float64, nullable: true, ...`
        // so we just extract bits we need
        //
        // In general type handling is very optimistic
        // at the moment, but good enough for poc
        DataType::List(f) => f.data_type().clone(),
        r => r.clone(),
    }
}

pub fn configure_context() -> SessionContext {
    let runtime_environment = RuntimeEnv::new(RuntimeConfig::new()).unwrap();

    let session_config = SessionConfig::new()
        .with_information_schema(true)
        .with_option_extension(TorchConfig::default());

    let state = SessionStateBuilder::new()
        .with_config(session_config)
        .with_runtime_env(runtime_environment.into())
        .with_default_features()
        .with_function_factory(Some(Arc::new(TorchFunctionFactory::default())))
        .build();

    let ctx = SessionContext::new_with_state(state);

    ctx.register_udf(ScalarUDF::from(crate::ArgMax::new()));

    ctx
}
#[cfg(test)]
mod test {
    use datafusion::{assert_batches_eq, error::Result, execution::object_store::ObjectStoreUrl};
    use object_store::aws::AmazonS3Builder;

    #[tokio::test]
    async fn e2e() -> Result<()> {
        let ctx = crate::configure_context();

        let sql = r#"
        CREATE EXTERNAL TABLE iris 
        STORED AS PARQUET 
        LOCATION 'data/iris.snappy.parquet';
        "#;

        ctx.sql(sql).await?.collect().await?;

        let sql = r#"
        CREATE FUNCTION iris(FLOAT[])
        RETURNS FLOAT[]
        LANGUAGE TORCH
        AS 'model/iris.spt'
        "#;

        ctx.sql(sql).await?.collect().await?;

        let sql = r#"
        SELECT 
        argmax(iris(features)) as f_inferred, 
        argmax(iris([sl, sw, pl, pw])) as inferred
        FROM iris 
        LIMIT 15
        "#;

        let expected = vec![
            "+------------+----------+",
            "| f_inferred | inferred |",
            "+------------+----------+",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 1          | 1        |",
            "+------------+----------+",
        ];

        let result = ctx.sql(sql).await?.collect().await?;
        assert_batches_eq!(expected, &result);
        Ok(())
    }

    #[ignore = "too lazy to setup testcontainer for this test"]
    #[tokio::test]
    async fn e2e_s3() -> Result<()> {
        let ctx = crate::configure_context();
        ctx.state_ref().write().runtime_env().register_object_store(
            ObjectStoreUrl::parse("s3://modelrepo").unwrap().as_ref(),
            std::sync::Arc::new(
                AmazonS3Builder::from_env()
                    .with_bucket_name("modelrepo")
                    .with_allow_http(true)
                    .with_access_key_id("TxT2BsMsVSWgUikQm4mq")
                    .with_secret_access_key("FWFvBnRrxy2Up42LnA1QzGtmC8b6VZAhE7Tbt45R")
                    .with_endpoint("http://host.docker.internal:9000")
                    .with_region("local")
                    .build()
                    .unwrap(),
            ),
        );
        let sql = r#"
        CREATE EXTERNAL TABLE iris 
        STORED AS PARQUET 
        LOCATION 'data/iris.snappy.parquet';
        "#;

        ctx.sql(sql).await?.collect().await?;

        let sql = r#"
        CREATE FUNCTION iris(FLOAT[])
        RETURNS FLOAT[]
        LANGUAGE TORCH
        AS 's3://modelrepo/iris.spt'
        "#;

        ctx.sql(sql).await?.collect().await?;

        let sql = r#"
        SELECT 
        argmax(iris(features)) as f_inferred, 
        argmax(iris([sl, sw, pl, pw])) as inferred
        FROM iris 
        LIMIT 15
        "#;

        let expected = vec![
            "+------------+----------+",
            "| f_inferred | inferred |",
            "+------------+----------+",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 0          | 0        |",
            "| 1          | 1        |",
            "| 1          | 1        |",
            "+------------+----------+",
        ];

        let result = ctx.sql(sql).await?.collect().await?;
        assert_batches_eq!(expected, &result);
        Ok(())
    }
}
