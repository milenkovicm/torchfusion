# TorchFusion

Torchfusion is an very opiniated torch and datafusion integration, implemented to demonstrate datafusion `FunctionFactory` functionality merge request ([arrow-datafusion/pull#9333](https://github.com/apache/arrow-datafusion/pull/9333)). It has not been envisaged as a activly maintained library.

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
    sl, sw, pl, pw,
    features, 
    argmax(iris(features)) as f_inferred, 
    argmax(iris([sl, sw, pl, pw])) as inferred, 
    label as label
    from iris 
    limit 50
"#;

ctx.sql(sql).await?.show().await?;

Ok(())
```

```txt
++
++
+-----+-----+-----+-----+----------------------+------------+----------+-------+
| sl  | sw  | pl  | pw  | features             | f_inferred | inferred | label |
+-----+-----+-----+-----+----------------------+------------+----------+-------+
| 4.4 | 3.0 | 1.3 | 0.2 | [4.4, 3.0, 1.3, 0.2] | 0          | 0        | 0     |
| 5.5 | 4.2 | 1.4 | 0.2 | [5.5, 4.2, 1.4, 0.2] | 0          | 0        | 0     |
| 5.7 | 2.9 | 4.2 | 1.3 | [5.7, 2.9, 4.2, 1.3] | 1          | 1        | 1     |
| 5.8 | 2.7 | 3.9 | 1.2 | [5.8, 2.7, 3.9, 1.2] | 1          | 1        | 1     |
| 5.9 | 3.0 | 4.2 | 1.5 | [5.9, 3.0, 4.2, 1.5] | 1          | 1        | 1     |
| 5.9 | 3.0 | 5.1 | 1.8 | [5.9, 3.0, 5.1, 1.8] | 2          | 2        | 2     |
| 6.1 | 2.8 | 4.0 | 1.3 | [6.1, 2.8, 4.0, 1.3] | 1          | 1        | 1     |
| 6.1 | 2.8 | 4.7 | 1.2 | [6.1, 2.8, 4.7, 1.2] | 1          | 1        | 1     |
| 6.2 | 2.8 | 4.8 | 1.8 | [6.2, 2.8, 4.8, 1.8] | 2          | 2        | 2     |
| 6.4 | 2.7 | 5.3 | 1.9 | [6.4, 2.7, 5.3, 1.9] | 2          | 2        | 2     |
| 6.4 | 3.2 | 4.5 | 1.5 | [6.4, 3.2, 4.5, 1.5] | 1          | 1        | 1     |
+-----+-----+-----+-----+----------------------+------------+----------+-------+
 *  Terminal will be reused by tasks, press any key to close it. 

```