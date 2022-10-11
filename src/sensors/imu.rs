type Vec3<T> = [T; 3];

pub struct IMUUpdate {
    pub timestamp: f64,
    pub acceleration: Vec3<f64>,
    pub rotation: Vec3<f64>,
}
