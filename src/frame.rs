#[derive(Default)]
pub struct Frame {
    pub timestamp: f64,
    pub imu_bias: Bias,
}

impl Frame {
    pub fn is_set(&self) -> bool {
        todo!()
    }
}

#[derive(Default, Clone)]
pub struct KeyFrame {
    pub imu_bias: Bias,
}

#[derive(Default, Clone)]
pub struct Bias {}
