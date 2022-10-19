/// Representation of a pixel point on an image
#[derive(Clone, Default)]
pub struct Keypoint {
    pub x: u32,
    pub y: u32,
}

#[derive(Default)]
pub struct Frame<Feat> {
    pub features: Vec<Feat>,
    pub timestamp: f64,
}

#[derive(Default)]
pub struct KeyframeDatabase<F> {
    pub key_frames: Vec<Frame<F>>,
}
