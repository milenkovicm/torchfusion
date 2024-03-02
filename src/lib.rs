use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{
            ArrayBuilder, ArrayRef, Float32Builder, Float64Builder, GenericListArray, ListArray,
            PrimitiveArray, PrimitiveBuilder, UInt32Builder,
        },
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::{ArrowPrimitiveType, DataType, Field},
        error::Result,
    },
    error::DataFusionError,
    execution::{
        config::SessionConfig,
        context::{FunctionFactory, RegisterFunction},
    },
    logical_expr::{
        create_udf, ColumnarValue, CreateFunction, DefinitionStatement, ScalarUDF, Volatility,
    },
};

use tch::{nn::Module, CModule};

pub struct TorchFunctionFactory {}

#[async_trait::async_trait]
impl FunctionFactory for TorchFunctionFactory {
    async fn create(
        &self,
        _state: &SessionConfig,
        statement: CreateFunction,
    ) -> datafusion::error::Result<RegisterFunction> {
        let model_name = statement.name;

        let data_type = statement
            .args
            .map(|a| {
                a.first()
                    .map(|r| r.data_type.clone())
                    .unwrap_or(DataType::Float32)
            })
            .unwrap_or(DataType::Float32);

        let data_type = match data_type {
            // We're interested in array type not the array.
            // There is discrepancy between array type defined by create function
            // `List(Field { name: \"field\", data_type: Float32, nullable:  ...``
            // and arry type defined by create array operation
            //`[List(Field { name: \"item\", data_type: Float64, nullable: true, ...`
            // so we just extract bits we need
            //
            // In general type handling is very optimistic
            // at the moment, but good enough for poc
            DataType::List(f) => f.data_type().clone(),
            r => r,
        };

        let model_file = match statement.params.as_ {
            Some(DefinitionStatement::DoubleDollarDef(s)) => s,
            Some(DefinitionStatement::SingleQuotedDef(s)) => s,
            _ => format!("model/{}.spt", model_name),
        };

        let model_udf = load_torch_model(&model_name, &model_file, data_type)?;

        Ok(RegisterFunction::Scalar(Arc::new(model_udf)))
    }
}

/// very opiniated torch model integration
/// it has been implemented to demonstrate integration
///
/// do not use it for anything
pub fn load_torch_model(model_name: &str, model_file: &str, dtype: DataType) -> Result<ScalarUDF> {
    let type_filed = Arc::new(Field::new("item", dtype.clone(), false));
    let type_item = DataType::List(type_filed.clone());
    let type_args = vec![type_item.clone()];
    // we should handle return type as defined in function declaration
    let type_return = Arc::new(type_item.clone());

    //
    //
    //

    let mut model =
        tch::CModule::load(model_file).map_err(|e| DataFusionError::Execution(e.to_string()))?;
    model
        .f_set_eval()
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    //
    //
    //

    let model_proxy = Arc::new(move |args: &[ColumnarValue]| {
        let args = ColumnarValue::values_to_arrays(args)?;
        let features = datafusion::common::cast::as_list_array(&args[0])?;

        match dtype.clone() {
            DataType::Float32 => {
                let values = datafusion::common::cast::as_float32_array(features.values())?;
                let offsets = features.offsets();

                let array = _call_model(
                    Float32Builder::new(),
                    values,
                    offsets,
                    &model,
                    type_filed.clone(),
                )?;
                Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
            }

            DataType::Float64 => {
                let values = datafusion::common::cast::as_float64_array(features.values())?;
                let offsets = features.offsets();

                let array = _call_model(
                    Float64Builder::new(),
                    values,
                    offsets,
                    &model,
                    type_filed.clone(),
                )?;
                Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
            }
            m => Err(datafusion::error::DataFusionError::Execution(format!(
                "no implementation for type: {}",
                m
            ))),
        }
    });

    let model_udf = create_udf(
        &format!("torch.{}", model_name),
        type_args,
        type_return,
        Volatility::Volatile,
        model_proxy,
    );

    Ok(model_udf)
}

// this should implement better batching , cover corner cases and so on ...
fn _call_model<T: ArrowPrimitiveType>(
    mut result: PrimitiveBuilder<T>,
    values: &PrimitiveArray<T>,
    offsets: &OffsetBuffer<i32>,
    model: &CModule,
    type_filed: Arc<Field>,
) -> Result<GenericListArray<i32>>
where
    <T as ArrowPrimitiveType>::Native: tch::kind::Element,
{
    let mut result_offsets: Vec<i32> = vec![];
    result_offsets.push(0);

    for o in offsets.windows(2) {
        let start = o[0] as usize;
        let end = o[1] as usize;
        // batching should be provided,
        // device selection ...
        let current: &[T::Native] = &values.values()[start..end];
        let tensor = tch::Tensor::from_slice(current);
        let logits = model.forward(&tensor);

        // can we use tensor to convert logits type to result type?
        let logits = Vec::<T::Native>::try_from(logits)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        result.append_slice(&logits[..]);
        result_offsets.push(result.len() as i32);
    }

    let array = ListArray::new(
        type_filed.clone(),
        OffsetBuffer::new(ScalarBuffer::from(result_offsets)),
        Arc::new(result.finish()),
        None,
    );

    Ok(array)
}
/// there is probably better implementation of argmax
pub fn f32_argmax_udf() -> ScalarUDF {
    let f = Arc::new(move |args: &[ColumnarValue]| {
        let args = ColumnarValue::values_to_arrays(args)?;
        let features = datafusion::common::cast::as_list_array(&args[0])?;

        let values = datafusion::common::cast::as_float32_array(features.values())?;
        let offsets = features.offsets();

        let mut result = UInt32Builder::new();
        offsets.windows(2).for_each(|o| {
            let start = o[0] as usize;
            let end = o[1] as usize;

            let current = &values.values()[start..end];

            result.append_option(f32_argmax(current));
        });

        Ok(ColumnarValue::from(Arc::new(result.finish()) as ArrayRef))
    });

    let arg_type = DataType::List(Arc::new(Field::new("item", DataType::Float32, false)));
    let args = vec![arg_type.clone()];

    create_udf(
        "f32_argmax",
        args,
        Arc::new(DataType::UInt32),
        Volatility::Immutable,
        f,
    )
}

fn f32_argmax(values: &[f32]) -> Option<u32> {
    let result = values
        .iter()
        .enumerate()
        .fold(None, |a, (pos, value)| match a {
            None => Some((pos, value)),
            Some((_, a_value)) if value > a_value => Some((pos, value)),
            Some(a) => Some(a),
        });

    result.map(|(pos, _)| pos as u32)
}

mod test {

    #[test]
    fn arg_max_test() {
        assert_eq!(Some(1), crate::f32_argmax(&vec![1.0, 3.0, 2.0]))
    }
}
