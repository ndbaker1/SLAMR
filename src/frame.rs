use image::{ImageBuffer, Luma};
use imageproc::corners::{corners_fast9, Corner};

/// Representation of a pixel point on an image
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

/// Feature object which holds a coordinate/pixel on a page
/// and tries to handle a generic descriptor
pub struct Feature<D = u128> {
    pub keypoint: KeyPoint,
    pub descriptor: D,
}

impl Feature {
    /// Uses FAST (Features from Accelerated Segment Test) as a keypoint detector for features like corners in a grayscale image,
    /// then applies the BRIEF (Binary Robust Independent Elementary Features) to compute descriptors for these keypoints.
    pub fn from_fast_and_brief_128(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Self> {
        // apply a guassion blur to the image for computing BRIEF descriptors,
        // that way the image is not overly sesnsitive to high frequency noise.
        //
        // using a kernel value of 2 indicated by reference:
        // https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6
        let smoothed_image = imageproc::filter::gaussian_blur_f32(&image, 2.0);

        corners_fast9(image, 20)
            .into_iter()
            .map(|Corner { x, y, .. }| {
                let keypoint = KeyPoint { x, y };
                let descriptor = compute_brief_128(&keypoint, &smoothed_image);
                Feature {
                    keypoint,
                    descriptor,
                }
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Frame {
    pub timestamp: f64,
    pub features: Vec<Feature>,
}

impl Frame {
    pub fn is_set(&self) -> bool {
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

pub fn compute_brief_128(keypoint: &KeyPoint, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> u128 {
    5
}
