use argminmax::ArgMinMax;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Float16Array, PrimitiveArray, UInt32Builder},
        buffer::OffsetBuffer,
        datatypes::{ArrowPrimitiveType, DataType, Field},
    },
    common::{downcast_value, internal_err},
    error::{DataFusionError, Result},
    logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility},
};
use std::{any::Any, fmt::Debug, sync::Arc};

#[derive(Debug, Clone)]
pub struct ArgMax {
    signature: Signature,
}

impl Default for ArgMax {
    fn default() -> Self {
        Self::new()
    }
}

impl ArgMax {
    pub fn new() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![
                    DataType::Float32,
                    DataType::Float64,
                    DataType::Int32,
                    DataType::Int64,
                ]
                .into_iter()
                .map(|t| DataType::List(Arc::new(Field::new("item", t, false))))
                .collect::<Vec<_>>(),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for ArgMax {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "argmax"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        let args = ColumnarValue::values_to_arrays(args)?;
        let features = datafusion::common::cast::as_list_array(&args[0])?;

        let data_type = match features.data_type() {
            DataType::List(f) => f.data_type(),
            t => internal_err!("Argument should be a list {t}")?,
        };
        let offsets = features.offsets();
        match data_type {
            DataType::Float16 => {
                let values = downcast_value!(features.values(), Float16Array);
                Self::find_max(values, offsets)
            }
            DataType::Float32 => {
                let values = datafusion::common::cast::as_float32_array(features.values())?;
                Self::find_max(values, offsets)
            }
            DataType::Float64 => {
                let values = datafusion::common::cast::as_float64_array(features.values())?;
                Self::find_max(values, offsets)
            }
            DataType::Int32 => {
                let values = datafusion::common::cast::as_int32_array(features.values())?;
                Self::find_max(values, offsets)
            }
            DataType::Int64 => {
                let values = datafusion::common::cast::as_int64_array(features.values())?;
                Self::find_max(values, offsets)
            }
            t => internal_err!("Unsupported type {t}")?,
        }
    }

    fn aliases(&self) -> &[String] {
        &[]
    }
}

impl ArgMax {
    fn find_max<T: ArrowPrimitiveType>(
        values: &PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
    ) -> Result<ColumnarValue>
    where
        for<'a> &'a [T::Native]: ArgMinMax,
    {
        let mut result = UInt32Builder::new();
        offsets.windows(2).for_each(|o| {
            let start = o[0] as usize;
            let end = o[1] as usize;

            let current: &[T::Native] = &values.values()[start..end];
            result.append_value(current.argmax() as u32);
        });

        Ok(ColumnarValue::from(Arc::new(result.finish()) as ArrayRef))
    }
}
