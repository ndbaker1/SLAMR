#[derive(Default)]
pub struct Frame {
    pub timestamp: f64,
    pub imu_bias: Bias,
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
