pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

pub type Descriptor = u32;

pub type Feature = (KeyPoint, Descriptor);

#[derive(Default)]
pub struct Frame {
    pub timestamp: f64,
    pub imu_bias: Bias,
    pub features: Vec<Feature>,
}

impl Frame {
    pub fn is_set(&self) -> bool {
        todo!()
    }

    pub fn extract_features() {
        todo!()
    }
}

#[derive(Default)]
pub struct KeyFrame {
    pub frame: Frame,
}

#[derive(Default, Clone)]
pub struct Bias {}

#[derive(Default)]
pub struct KeyframeDatabase {
    key_frames: Vec<KeyFrame>,
}
