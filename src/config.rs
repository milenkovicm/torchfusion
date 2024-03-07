use datafusion::error::Result;
use datafusion::{
    config::{ConfigEntry, ConfigExtension, ExtensionOptions},
    error::DataFusionError,
};
use tch::Device;

#[derive(Debug, Clone)]
pub struct TorchConfig {
    device: Device,
    cuda_device: usize,
    model_non_blocking: bool,
    batch_size: usize,
}

impl Default for TorchConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            cuda_device: 0,
            model_non_blocking: false,
            batch_size: 1,
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
            "model_non_blocking" => {
                self.model_non_blocking = value.parse().map_err(|_| {
                    DataFusionError::Configuration("non blocing should be boolean".to_string())
                })?
            }
            "batch_size" => {
                self.batch_size = value.parse().map_err(|_| {
                    DataFusionError::Configuration("batch size not correct".to_string())
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
                key: format!("{}.device", Self::PREFIX),
                value: Some(format!("{:?}", self.device)),
                description: "Device to run model on. Valid values 'cpu', 'cuda', 'mps', 'vulkan'. Default: 'cpu' ",
            },
            ConfigEntry {
                key: format!("{}.cuda_device", Self::PREFIX),
                value: Some(format!("{}", self.cuda_device)),
                description: "Cuda device to use. Valid value positive integer. Default: 0",
            },
            ConfigEntry {
                key: format!("{}.model_non_blocking", Self::PREFIX),
                value: Some(format!("{}", self.model_non_blocking)),
                description: "Non-blocking memory transfer. Valid value boolean. Default: false",
            },
            ConfigEntry {
                key: format!("{}.batch_size", Self::PREFIX),
                value: Some(format!("{}", self.batch_size)),
                description: "Batch size to be used. Valid value positive non-zero integers. Default: 1",
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
                "Device '{device}' not supported"
            )))?,
        }
    }
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn model_non_blocking(&self) -> bool {
        self.model_non_blocking
    }
}

impl ConfigExtension for TorchConfig {
    const PREFIX: &'static str = "torchfusion";
}
