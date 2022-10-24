#[derive(Default)]
pub struct Frame<Feat> {
    pub features: Vec<Feat>,
    pub timestamp: f64,
}

#[derive(Default)]
pub struct KeyframeDatabase<F> {
    pub key_frames: Vec<Frame<F>>,
}
