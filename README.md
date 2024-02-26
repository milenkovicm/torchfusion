# TorchFusion

Torchfusion is an very opiniated torch and datafusion integration, implemented to demonstrate datafusion `FunctionFactory` functionality merge request. It has not been envisaged as a activly maintained library.

## How to use

A torch model can be defined as and SQL UDF definition:

```sql
CREATE FUNCTION iris(FLOAT[])
RETURNS FLOAT[]
LANGUAGE TORCH
AS '/models/iris.pt'
```

Where function parameter defines type of input array and return type defines type of return array. 
Return statement points to locaton where scripted model file is located.

or, something which is not implemented in this example, referencing a model in MlFlow repository:

```sql
CREATE FUNCTION iris(FLOAT[])
RETURNS FLOAT[]
LANGUAGE TORCH
AS 'models:/iris@champion'
```

so overall flow should be:

```rust
let runtime_config = RuntimeConfig::new();
let runtime_environment = RuntimeEnv::new(runtime_config).unwrap();
let session_config =
    SessionConfig::new().set_str("datafusion.sql_parser.dialect", "PostgreSQL");
let state = SessionState::new_with_config_rt(session_config, Arc::new(runtime_environment))
    .with_function_factory(Arc::new(TorchFunctionFactory {}));

let ctx = SessionContext::new_with_state(state);
ctx.register_udf(torchfusion::f32_argmax_udf());

ctx.register_parquet("iris", "data/iris.snappy.parquet", Default::default())
    .await
    .expect("table to be loaded");

// we define a torch model to use
let sql = r#"
CREATE FUNCTION iris(FLOAT[])
RETURNS FLOAT[]
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
```


## What else can be implemented using `FunctionFactory`

`FunctionFactory` is an extension point, a SQL defined functions like

```sql
CREATE FUNCTION add(BIGINT, BIGINT)
RETURNS BIGINT
LANGUAGE SQL
RETURN $1 + $2
```

would be easy to implement. In addition to `FunctionFactory` a `AnalyzerRule` should be implemented, which can alter logical plan.
