use std::{any::Any, fmt::Debug, marker::PhantomData, sync::Arc};

use datafusion::{
    arrow::{
        array::{Array, ArrayBuilder, ArrayRef, ListArray, PrimitiveArray, PrimitiveBuilder},
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::{ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type},
    },
    common::{downcast_value, exec_err},
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility},
};
use tch::{nn::Module, CModule, Device, Kind};

/// very opiniated torch model integration
/// it has been implemented to demonstrate integration 
/// with datafusion as user defined function.
/// 
/// do not use it for anything

pub fn load_torch_model(
    model_name: &str,
    model_file: &str,
    device: Device,
    // we need this two guys
    input_type: DataType,
    return_type: DataType,
) -> Result<ScalarUDF> {
    match (input_type, return_type) {
        (DataType::Float32, DataType::Float32) => {
            let model_udf = TorchUdf::<Float32Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }

        (DataType::Float64, DataType::Float32) => {
            let model_udf = TorchUdf::<Float64Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }

        (DataType::Float64, DataType::Float64) => {
            let model_udf = TorchUdf::<Float64Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }
        // we can add few more later
        (i, r) => exec_err!(
            "data type combination not supported args: {}, return: {}",
            i,
            r
        )?,
    }

    // let model_udf = TorchUdf::<Float32Type, Float32Type>::new_from_file(
    //     model_name.to_string(),
    //     model_file,
    //     device,
    // )?;

    //Ok(ScalarUDF::from(model_udf))
}

#[derive(Debug)]
struct TorchUdf<I: ArrowPrimitiveType + Debug, R: ArrowPrimitiveType + Debug> {
    name: String,
    device: Device,
    model: CModule,
    signature: Signature,
    /// type of expected input list (not DataType::List)
    //#[allow(dead_code)]
    //defined_input_type: DataType,
    /// type of expected result list (not DataType::List)
    //#[allow(dead_code)]
    //defined_return_type: DataType,
    return_type_filed: Arc<Field>,
    phantom_i: PhantomData<I>,
    phantom_r: PhantomData<R>,
}

impl<I: ArrowPrimitiveType + Debug, R: ArrowPrimitiveType + Debug> TorchUdf<I, R> {
    fn new_from_file(
        name: String,
        model_file: &str,
        device: Device,
        //input_type: DataType,
        //return_type: DataType,
    ) -> Result<Self> {
        //R::DATA_TYPE
        let kind = Self::to_torch_type(&I::DATA_TYPE.clone())?;
        let model = Self::load_model(model_file, device, kind)?;

        Ok(Self::new(name, model, device))
    }

    fn new(
        name: String,
        model: CModule,
        device: Device,
        //defined_input_type: DataType,
        //defined_return_type: DataType,
    ) -> Self {
        let return_type_filed = Arc::new(Field::new("item", R::DATA_TYPE.clone(), false));
        Self {
            // TODO: is uniform signature required
            //       no it is not, revert back to exact
            signature: Signature::exact(
                vec![DataType::List(Arc::new(Field::new(
                    "item",
                    I::DATA_TYPE.clone(),
                    false,
                )))],
                Volatility::Immutable,
            ),
            name,
            device,
            model,
            return_type_filed,
            phantom_i: PhantomData,
            phantom_r: PhantomData,
        }
    }

    fn load_model(model_file: &str, device: Device, kind: Kind) -> Result<CModule> {
        let mut model = tch::CModule::load(model_file)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        model.to(device, kind, false);
        model
            .f_set_eval()
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        Ok(model)
    }
}

impl<R: ArrowPrimitiveType + Debug + Send + Sync, I: ArrowPrimitiveType + Debug + Send + Sync>
    ScalarUDFImpl for TorchUdf<R, I>
where
    <R as ArrowPrimitiveType>::Native: tch::kind::Element,
    <I as ArrowPrimitiveType>::Native: tch::kind::Element,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(self.return_type_filed.clone()))
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let args = ColumnarValue::values_to_arrays(args)?;
        let features = datafusion::common::cast::as_list_array(&args[0])?;

        let offsets = features.offsets();

        let (result_offsets, values) = {
            let values = downcast_value!(features.values(), PrimitiveArray, I);
            Self::call_model(
                PrimitiveBuilder::<R>::new(),
                values,
                offsets,
                &self.model,
                self.device,
            )?
        };
        let array = ListArray::new(self.return_type_filed.clone(), result_offsets, values, None);

        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}

impl<R: ArrowPrimitiveType + Debug, I: ArrowPrimitiveType + Debug> TorchUdf<R, I> {
    #[inline]
    fn call_model<T: ArrowPrimitiveType>(
        mut result: PrimitiveBuilder<R>,
        values: &PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
        model: &CModule,
        device: Device,
    ) -> Result<(
        OffsetBuffer<i32>,
        Arc<(dyn datafusion::arrow::array::Array + 'static)>,
    )>
    where
        <T as ArrowPrimitiveType>::Native: tch::kind::Element,
        <R as ArrowPrimitiveType>::Native: tch::kind::Element,
    {
        let mut result_offsets: Vec<i32> = vec![];
        result_offsets.push(0);

        for o in offsets.windows(2) {
            let start = o[0] as usize;
            let end = o[1] as usize;
            // batching should be provided,
            // device selection ...
            let current: &[T::Native] = &values.values()[start..end];
            let tensor = tch::Tensor::from_slice(current).to(device);
            let logits = model.forward(&tensor);

            // can we use tensor to convert logits type to result type?
            let logits = Vec::<R::Native>::try_from(logits)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            result.append_slice(&logits[..]);
            result_offsets.push(result.len() as i32);
        }

        Ok((
            OffsetBuffer::new(ScalarBuffer::from(result_offsets)),
            Arc::new(result.finish()) as ArrayRef,
        ))
    }

    // fn item_type(dtype: &DataType) -> Result<&DataType> {
    //     match dtype {
    //         DataType::List(f) => Ok(f.data_type()),
    //         t => exec_err!("Input data type not supported {t}, expecting a list")?,
    //     }
    // }

    fn to_torch_type(dtype: &DataType) -> Result<Kind> {
        match &dtype {
            DataType::Boolean => Ok(tch::Kind::Bool),
            DataType::Int16 => Ok(tch::Kind::Int16),
            DataType::Int32 => Ok(tch::Kind::Int),
            DataType::Int64 => Ok(tch::Kind::Int64),
            DataType::Float16 => Ok(tch::Kind::BFloat16),
            DataType::Float32 => Ok(tch::Kind::Float),
            DataType::Float64 => Ok(tch::Kind::Double),
            t => exec_err!("type not coverd: {t}")?,
        }
    }
}
