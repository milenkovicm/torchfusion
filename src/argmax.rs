use std::{any::Any, fmt::Debug, sync::Arc};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, PrimitiveArray, UInt32Builder},
        buffer::OffsetBuffer,
        datatypes::{ArrowPrimitiveType, DataType, Field},
    },
    common::internal_err,
    error::Result,
    logical_expr::{ColumnarValue, FuncMonotonicity, ScalarUDFImpl, Signature, Volatility},
};

#[derive(Debug, Clone)]
pub struct ArgMax {
    signature: Signature,
}

impl Default for ArgMax {
    fn default() -> Self {
        Self::new()
    }
}

// replace with https://docs.rs/argminmax/latest/argminmax/
// https://github.com/jvdd/argminmax
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

        match data_type {
            DataType::Float32 => {
                let values = datafusion::common::cast::as_float32_array(features.values())?;
                let offsets = features.offsets();
                Self::find_max(values, offsets)
            }
            DataType::Float64 => {
                let values = datafusion::common::cast::as_float32_array(features.values())?;
                let offsets = features.offsets();
                Self::find_max(values, offsets)
            }
            DataType::Int32 => {
                let values = datafusion::common::cast::as_int32_array(features.values())?;
                let offsets = features.offsets();
                Self::find_max(values, offsets)
            }
            DataType::Int64 => {
                let values = datafusion::common::cast::as_int64_array(features.values())?;
                let offsets = features.offsets();
                Self::find_max(values, offsets)
            }
            t => internal_err!("Unsuported type {t}")?,
        }
    }

    fn aliases(&self) -> &[String] {
        &[]
    }

    fn monotonicity(&self) -> Result<Option<FuncMonotonicity>> {
        Ok(None)
    }
}

impl ArgMax {
    fn find_max<T: ArrowPrimitiveType>(
        values: &PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
    ) -> Result<ColumnarValue> {
        let mut result = UInt32Builder::new();
        offsets.windows(2).for_each(|o| {
            let start = o[0] as usize;
            let end = o[1] as usize;

            let current = &values.values()[start..end];

            result.append_option(Self::argmax(current));
        });

        Ok(ColumnarValue::from(Arc::new(result.finish()) as ArrayRef))
    }

    fn argmax<T: std::cmp::PartialOrd>(values: &[T]) -> Option<u32> {
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
}

mod test {

    #[test]
    fn arg_max_test() {
        assert_eq!(Some(1), crate::argmax::ArgMax::argmax(&vec![1.0, 3.0, 2.0]))
    }
}
