use std::{any::Any, sync::Arc};

use datafusion::{
    arrow::{
        array::{
            Array, ArrayBuilder, ArrayRef, Float16Array, Float32Builder, Float64Builder, ListArray,
            PrimitiveArray, PrimitiveBuilder, UInt32Builder,
        },
        buffer::{OffsetBuffer, ScalarBuffer},
        datatypes::{ArrowPrimitiveType, DataType, Field},
    },
    common::{exec_err, internal_err},
    config::{ConfigEntry, ConfigExtension, ExtensionOptions},
    error::{DataFusionError, Result},
    execution::{
        config::SessionConfig,
        context::{FunctionFactory, RegisterFunction, SessionContext, SessionState},
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    logical_expr::{
        ColumnarValue, CreateFunction, DefinitionStatement, FuncMonotonicity, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility,
    },
};

use log::debug;
use tch::{nn::Module, CModule, Device, Kind};

pub struct TorchFunctionFactory {}

#[async_trait::async_trait]
impl FunctionFactory for TorchFunctionFactory {
    async fn create(
        &self,
        state: &SessionConfig,
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

        let arg_data_type = find_item_type(&arg_data_type);

        let return_data_type = statement
            .return_type
            .map(|t| find_item_type(&t))
            .unwrap_or(arg_data_type.clone());

        let model_file = match statement.params.as_ {
            Some(DefinitionStatement::DoubleDollarDef(s)) => s,
            Some(DefinitionStatement::SingleQuotedDef(s)) => s,
            _ => format!("model/{}.spt", model_name),
        };
        let config = state
            .options()
            .extensions
            .get::<TorchConfig>()
            .expect("torch configuration to be configured");

        // same device will be used untill function is dropped
        let device = config.device;
        let model_udf = load_torch_model(
            &model_name,
            &model_file,
            device,
            arg_data_type,
            return_data_type,
        )?;
        debug!("reistering function: [{:?}]", model_udf);

        Ok(RegisterFunction::Scalar(Arc::new(model_udf)))
    }
}

fn find_item_type(dtype: &DataType) -> DataType {
    match dtype {
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
        r => r.clone(),
    }
}

/// very opiniated torch model integration
/// it has been implemented to demonstrate integration
///
/// do not use it for anything
pub fn load_torch_model(
    model_name: &str,
    model_file: &str,
    device: Device,
    input_type: DataType,
    return_type: DataType,
) -> Result<ScalarUDF> {
    let model_udf = TorchUdf::new_from_file(
        model_name.to_string(),
        model_file,
        device,
        input_type,
        return_type,
    )?;

    Ok(ScalarUDF::from(model_udf))
}

#[derive(Debug)]
struct TorchUdf {
    name: String,
    device: Device,
    model: CModule,
    signature: Signature,
    /// type of expected input list (not DataType::List)
    #[allow(dead_code)]
    defined_input_type: DataType,
    /// type of expected result list (not DataType::List)
    #[allow(dead_code)]
    defined_return_type: DataType,
    return_type_filed: Arc<Field>,
}

impl TorchUdf {
    fn new_from_file(
        name: String,
        model_file: &str,
        device: Device,
        input_type: DataType,
        return_type: DataType,
    ) -> Result<Self> {
        let kind = Self::to_torch_type(&input_type)?;
        let model = Self::load_model(model_file, device, kind)?;

        Ok(Self::new(name, model, device, input_type, return_type))
    }

    fn new(
        name: String,
        model: CModule,
        device: Device,
        defined_input_type: DataType,
        defined_return_type: DataType,
    ) -> Self {
        let return_type_filed = Arc::new(Field::new("item", defined_return_type.clone(), false));
        Self {
            signature: Signature::uniform(
                1,
                vec![
                    // DataType::Float16,
                    DataType::Float32,
                    DataType::Float64,
                    // DataType::Int16,
                    DataType::Int32,
                    DataType::Int64,
                ]
                .into_iter()
                .map(|t| DataType::List(Arc::new(Field::new("item", t, false))))
                .collect::<Vec<_>>(),
                Volatility::Immutable,
            ),
            name,
            device,
            model,
            defined_input_type,
            defined_return_type,
            return_type_filed,
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

impl ScalarUDFImpl for TorchUdf {
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

        let runtime_input_type = Self::item_type(features.data_type())?;

        let (result_offsets, values) = match runtime_input_type {
            DataType::Float16 => {
                let values = datafusion::common::downcast_value!(features.values(), Float16Array);

                Self::call_model(
                    Float32Builder::new(),
                    values,
                    offsets,
                    &self.model,
                    self.device,
                )?
            }
            DataType::Float32 => {
                let values = datafusion::common::cast::as_float32_array(features.values())?;

                Self::call_model(
                    Float32Builder::new(),
                    values,
                    offsets,
                    &self.model,
                    self.device,
                )?
            }

            DataType::Float64 => {
                let values = datafusion::common::cast::as_float64_array(features.values())?;

                Self::call_model(
                    Float64Builder::new(),
                    values,
                    offsets,
                    &self.model,
                    self.device,
                )?
            }
            t => exec_err!("Type not covered for infenrence: {t}")?,
        };

        let array = ListArray::new(self.return_type_filed.clone(), result_offsets, values, None);

        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    }
}

impl TorchUdf {
    #[inline]
    fn call_model<T: ArrowPrimitiveType, R: ArrowPrimitiveType>(
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

    fn item_type(dtype: &DataType) -> Result<&DataType> {
        match dtype {
            DataType::List(f) => Ok(f.data_type()),
            t => exec_err!("input data type not supported {t} expecting a list")?,
        }
    }

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

pub fn configure_context() -> SessionContext {
    let runtime_config = RuntimeConfig::new();
    let runtime_environment = RuntimeEnv::new(runtime_config).unwrap();

    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .extensions
        .insert(TorchConfig::default());
    let session_config = session_config.set_str("datafusion.sql_parser.dialect", "PostgreSQL");
    let state = SessionState::new_with_config_rt(session_config, Arc::new(runtime_environment))
        .with_function_factory(Arc::new(TorchFunctionFactory {}));
    let ctx = SessionContext::new_with_state(state);

    ctx.register_udf(ScalarUDF::from(ArgMax::new()));

    ctx
}

#[derive(Debug, Clone)]
pub struct TorchConfig {
    device: Device,
    cuda_device: usize,
}

impl Default for TorchConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            cuda_device: 0,
        }
    }
}

impl ExtensionOptions for TorchConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::error::Result<()> {
        match key.to_lowercase().as_str() {
            "device" => self.device = self.parse_device(value)?,
            "cuda_device" => {
                self.cuda_device = value.parse().map_err(|_| {
                    DataFusionError::Configuration("Cuda device id format not correct".to_string())
                })?
            }
            key => Err(DataFusionError::Configuration(format!(
                "No configuration key: {key}"
            )))?,
        }

        Ok(())
    }

    fn entries(&self) -> Vec<datafusion::config::ConfigEntry> {
        vec![
            ConfigEntry {
                key: "device".into(),
                value: Some(format!("{:?}", self.device)),
                description: "Device to run model on. Valid values 'cpu', 'cuda', 'mps', 'vulkan'. Default: 'cpu' ",
            },
            ConfigEntry {
                key: "cuda_device".into(),
                value: Some(format!("{}", self.cuda_device)),
                description: "Cuda device to use. Valid value positive integer. Default: 0",
            },
        ]
    }
}

impl TorchConfig {
    fn parse_device(&self, value: &str) -> Result<Device> {
        match value.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" if tch::utils::has_cuda() => Ok(Device::Cuda(self.cuda_device)),
            "mps" if tch::utils::has_mps() => Ok(Device::Mps),
            "vulkan" if tch::utils::has_vulkan() => Ok(Device::Vulkan),
            device => Err(DataFusionError::Configuration(format!(
                "No support for device: {device}"
            )))?,
        }
    }
    pub fn device(&self) -> Device {
        self.device
    }
}

impl ConfigExtension for TorchConfig {
    const PREFIX: &'static str = "torch";
}
#[derive(Debug, Clone)]
struct ArgMax {
    signature: Signature,
}

impl ArgMax {
    fn new() -> Self {
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
        assert_eq!(Some(1), crate::ArgMax::argmax(&vec![1.0, 3.0, 2.0]))
    }
}
