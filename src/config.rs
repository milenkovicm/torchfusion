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
                key: format!("{}.device", Self::PREFIX),
                value: Some(format!("{:?}", self.device)),
                description: "Device to run model on. Valid values 'cpu', 'cuda', 'mps', 'vulkan'. Default: 'cpu' ",
            },
            ConfigEntry {
                key: format!("{}.cuda_device", Self::PREFIX),
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
    const PREFIX: &'static str = "torchfusion";
}
