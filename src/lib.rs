use std::sync::Arc;

use datafusion::{
    arrow::datatypes::DataType,
    execution::{
        config::SessionConfig,
        context::{FunctionFactory, RegisterFunction, SessionContext, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    logical_expr::{CreateFunction, DefinitionStatement, ScalarUDF},
};
use log::debug;

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

        let model_file = match statement.params.as_ {
            Some(DefinitionStatement::DoubleDollarDef(s)) => s,
            Some(DefinitionStatement::SingleQuotedDef(s)) => s,
            _ => format!("model/{}.spt", model_name),
        };
        let config = state
            .config()
            .options()
            .extensions
            .get::<TorchConfig>()
            .expect("torch configuration to be configured");

        // same device will be used untill function is dropped
        let device = config.device();
        let non_blocking = config.model_non_blocking();
        let batch_size = config.batch_size();
        let model_udf = udf::load_torch_model(
            &model_name,
            &model_file,
            device,
            non_blocking,
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

    let state = SessionState::new_with_config_rt(session_config, Arc::new(runtime_environment))
        // register  factory configuration
        .with_function_factory(Arc::new(TorchFunctionFactory::default()));

    let ctx = SessionContext::new_with_state(state);

    ctx.register_udf(ScalarUDF::from(crate::ArgMax::new()));

    ctx
}
#[cfg(test)]
mod test {
    use datafusion::{assert_batches_eq, error::Result};

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
}
