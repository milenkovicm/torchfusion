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
use std::{any::Any, fmt::Debug, io::Read, marker::PhantomData, sync::Arc};
use tch::{nn::Module, CModule, Device, Tensor};

/// Very opiated torch model integration
/// it has been implemented to demonstrate integration
/// with datafusion as user defined function.
///
/// do not use it for anything

pub fn load_torch_model<F: Read>(
    model_name: &str,
    model_file: &mut F,
    device: Device,
    batch_size: usize,
    input_type: DataType,
    return_type: DataType,
) -> Result<ScalarUDF> {
    match (input_type, return_type) {
        (DataType::Float32, DataType::Float32) => {
            let model_udf = TorchUdf::<Float32Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
                batch_size,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }

        (DataType::Float64, DataType::Float32) => {
            let model_udf = TorchUdf::<Float64Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
                batch_size,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }

        (DataType::Float64, DataType::Float64) => {
            let model_udf = TorchUdf::<Float64Type, Float32Type>::new_from_file(
                model_name.to_string(),
                model_file,
                device,
                batch_size,
            )?;
            Ok(ScalarUDF::from(model_udf))
        }
        // we can add few more later
        (i, r) => exec_err!(
            "Data type combination not supported! args: {}, return: {}",
            i,
            r
        )?,
    }
}

#[derive(Debug)]
struct TorchUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
    batch_size: usize,
    name: String,
    device: Device,
    model: CModule,
    signature: Signature,
    return_type_filed: Arc<Field>,
    phantom_i: PhantomData<I>,
    phantom_r: PhantomData<R>,
}

impl<I, R> TorchUdf<I, R>
where
    I: ArrowPrimitiveType + Debug,
    R: ArrowPrimitiveType + Debug,
{
    fn new_from_file<F: Read>(
        name: String,
        model_file: &mut F,
        device: Device,
        batch_size: usize,
    ) -> Result<Self> {
        //let kind = Self::to_torch_type(&I::DATA_TYPE.clone())?;

        let model = Self::load_model(model_file, device)?;
        // TODO: at this point we can verify that input and output layers
        //       have the same types like function input and output
        //
        // model.named_parameters()
        //
        Ok(Self::new(name, model, device, batch_size))
    }

    fn new(name: String, model: CModule, device: Device, batch_size: usize) -> Self {
        let return_type_filed = Arc::new(Field::new("item", R::DATA_TYPE.clone(), false));
        Self {
            signature: Signature::exact(
                vec![DataType::List(Arc::new(Field::new(
                    "item",
                    I::DATA_TYPE.clone(),
                    false,
                )))],
                Volatility::Immutable,
            ),
            name,
            batch_size,
            device,
            model,
            return_type_filed,
            phantom_i: PhantomData,
            phantom_r: PhantomData,
        }
    }

    fn load_model<F: Read>(model_file: &mut F, device: Device) -> Result<CModule> {
        let mut model = tch::CModule::load_data_on_device(model_file, device)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        //model.to(device, kind, non_blocking);
        model
            .f_set_eval()
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        Ok(model)
    }
}

impl<R, I> ScalarUDFImpl for TorchUdf<R, I>
where
    R: ArrowPrimitiveType + Debug + Send + Sync,
    I: ArrowPrimitiveType + Debug + Send + Sync,
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
            Self::_call_model(
                PrimitiveBuilder::<R>::new(),
                values,
                offsets,
                &self.model,
                self.device,
                self.batch_size,
            )?
        };
        let array = ListArray::new(self.return_type_filed.clone(), result_offsets, values, None);

        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}

// consider creating iterator from this method
impl<R, I> TorchUdf<R, I>
where
    R: ArrowPrimitiveType + Debug,
    I: ArrowPrimitiveType + Debug,
{
    fn create_batched_tensor<T>(
        start_offset: usize,
        batch_size: usize,
        values: &PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
        device: Device,
    ) -> Option<(Tensor, usize, usize)>
    where
        T: ArrowPrimitiveType,
        <T as ArrowPrimitiveType>::Native: tch::kind::Element,
    {
        let end_offset = std::cmp::min(start_offset + batch_size, offsets.len() - 1);
        if end_offset <= start_offset {
            None
        } else {
            let index_start = offsets[start_offset] as usize;
            let index_end = offsets[end_offset] as usize;
            let total_items = end_offset - start_offset;

            let current: &[T::Native] = &values.values()[index_start..index_end];
            let tensor = tch::Tensor::from_slice(current)
                .view([total_items as i64, -1])
                .to(device);

            // total_items are not required as it can be extracted
            // from tensor. to extract it a bit of gymnastics is needed
            // so it was easier to just return it.
            // total_items are needed as we cant expect
            // to have batches aligned with inputs
            Some((tensor, end_offset, total_items))
        }
    }

    fn flatten_batched_tensor(
        tensor: Tensor,
        items: usize,
        result_offsets: &mut Vec<i32>,
        result: &mut PrimitiveBuilder<R>,
    ) -> Result<()>
    where
        <R as ArrowPrimitiveType>::Native: tch::kind::Element,
    {
        let start = result.len();

        // not sure if .contiguous() is needed
        let tensor = tensor.contiguous().view(-1);
        let logits = Vec::<R::Native>::try_from(tensor)
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        result.append_slice(&logits[..]);
        let end = result.len();
        let elements = (end - start) / items;

        // populate resulting offsets from result
        (1..=items).for_each(|i| result_offsets.push((start + i * elements) as i32));

        Ok(())
    }

    #[inline]
    fn _call_model<T>(
        mut result: PrimitiveBuilder<R>,
        values: &PrimitiveArray<T>,
        offsets: &OffsetBuffer<i32>,
        model: &CModule,
        device: Device,
        batch_size: usize,
    ) -> Result<(
        OffsetBuffer<i32>,
        Arc<(dyn datafusion::arrow::array::Array + 'static)>,
    )>
    where
        T: ArrowPrimitiveType,
        <T as ArrowPrimitiveType>::Native: tch::kind::Element,
        <R as ArrowPrimitiveType>::Native: tch::kind::Element,
    {
        let mut result_offsets: Vec<i32> = vec![];
        result_offsets.push(0);
        let mut start = 0;

        while let Some((tensor, next_start, no_items)) =
            // this would make more sense if we return iterator
            Self::create_batched_tensor(start, batch_size, values, offsets, device)
        {
            start = next_start;

            let logits = model.forward(&tensor);

            Self::flatten_batched_tensor(logits, no_items, &mut result_offsets, &mut result)?;
        }

        Ok((
            OffsetBuffer::new(ScalarBuffer::from(result_offsets)),
            Arc::new(result.finish()) as ArrayRef,
        ))
    }
}

#[cfg(test)]
mod test {
    use super::TorchUdf;
    use datafusion::arrow::{
        array::{Int32Array, Int32Builder},
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::Int32Type,
    };
    use tch::Tensor;

    #[test]
    fn should_create_tensor() {
        let values = Int32Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]);

        let offsets = vec![0, 2, 4, 6, 8, 10, 12, 14, 16];
        let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
        let device = tch::Device::Cpu;
        let batch_size = 3;
        let mut current_end = 0;

        if let Some((tensor, end, _total)) = TorchUdf::<Int32Type, Int32Type>::create_batched_tensor(
            current_end,
            batch_size,
            &values,
            &offsets,
            device,
        ) {
            current_end += end;
            assert_eq!(vec![3, 2], tensor.size());
        }

        if let Some((tensor, end, _total)) = TorchUdf::<Int32Type, Int32Type>::create_batched_tensor(
            current_end,
            batch_size,
            &values,
            &offsets,
            device,
        ) {
            current_end += end;
            assert_eq!(vec![3, 2], tensor.size());
        }

        if let Some((tensor, _end, _total)) =
            TorchUdf::<Int32Type, Int32Type>::create_batched_tensor(
                current_end,
                batch_size,
                &values,
                &offsets,
                device,
            )
        {
            assert_eq!(vec![2, 2], tensor.size());
        }

        let result = TorchUdf::<Int32Type, Int32Type>::create_batched_tensor(
            current_end,
            batch_size,
            &values,
            &offsets,
            device,
        );

        assert!(result.is_none());
    }

    #[test]
    fn should_flatten_tensor_0() {
        let mut result = Int32Builder::new();
        let mut result_offsets: Vec<i32> = vec![];
        result_offsets.push(0);

        let tensor =
            Tensor::from_slice2(&[[1, 2, 3], [11, 22, 33], [111, 222, 333], [1111, 2222, 3333]]);

        TorchUdf::<Int32Type, Int32Type>::flatten_batched_tensor(
            tensor,
            4,
            &mut result_offsets,
            &mut result,
        )
        .expect("no errors");
        let result = result.finish();

        assert_eq!(vec![0, 3, 6, 9, 12], result_offsets);
        assert_eq!(
            [1, 2, 3, 11, 22, 33, 111, 222, 333, 1111, 2222, 3333],
            result.values()[..]
        )
    }

    #[test]
    fn should_flatten_tensor_1() {
        let mut result_offsets: Vec<i32> = vec![];
        result_offsets.push(0);
        let tensor = Tensor::from_slice2(&[[1, 2], [11, 22]]);

        let mut result = Int32Builder::new();
        TorchUdf::<Int32Type, Int32Type>::flatten_batched_tensor(
            tensor,
            2,
            &mut result_offsets,
            &mut result,
        )
        .expect("no errors");
        let result = result.finish();

        assert_eq!(vec![0, 2, 4], result_offsets);
        assert_eq!([1, 2, 11, 22], result.values()[..])
    }
}
