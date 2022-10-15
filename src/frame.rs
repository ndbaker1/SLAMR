use image::{ImageBuffer, Luma};
use imageproc::corners::{corners_fast9, Corner};
use rand_distr::{Distribution, Normal};

/// Representation of a pixel point on an image
#[derive(Clone)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

/// Feature object which holds a coordinate/pixel on a page
/// and tries to handle a generic descriptor
#[derive(Clone)]
pub struct Feature<D = [u8; 16]> {
    pub keypoint: KeyPoint,
    pub descriptor: D,
}

impl Feature {
    /// Uses FAST (Features from Accelerated Segment Test) as a keypoint detector for features like corners in a grayscale image,
    /// then applies the BRIEF (Binary Robust Independent Elementary Features) to compute descriptors for these keypoints.
    pub fn from_fast_and_brief_128(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Self> {
        // using a kernel value of 2 indicated by reference:
        // https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6
        const GAUSSIAN_KERNEL_SIGMA: f32 = 2.0;
        const FAST_CORNERS_THESHOLD: u8 = 35;

        // apply a guassion blur to the image for computing BRIEF descriptors,
        // that way the image is not overly sesnsitive to high frequency noise.
        let smoothed_image = imageproc::filter::gaussian_blur_f32(&image, GAUSSIAN_KERNEL_SIGMA);

        corners_fast9(image, FAST_CORNERS_THESHOLD)
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

/// Compute BRIEF (Binary Robust Independent Elementary Features) on a given grayscale image given the target keypoint.
pub fn compute_brief_128(keypoint: &KeyPoint, image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> [u8; 16] {
    // assuming that normal distribution sampling is going to be bounded by (+-) 10 for now.
    // this results in patches that are 20 x 20 pixels
    let normal_dist: Normal<f64> = Normal::new(0 as _, 2 as _).unwrap();

    let mut brief_descriptor = [0; 16];
    for i in 0..16 {
        for _ in 0..u8::BITS {
            let (x1, y1) = (
                keypoint.x + normal_dist.sample(&mut rand::thread_rng()) as u32,
                keypoint.y + normal_dist.sample(&mut rand::thread_rng()) as u32,
            );
            let (x2, y2) = (
                keypoint.x + normal_dist.sample(&mut rand::thread_rng()) as u32,
                keypoint.y + normal_dist.sample(&mut rand::thread_rng()) as u32,
            );

            let (first, second) = match (
                image.get_pixel_checked(x1, y1),
                image.get_pixel_checked(x2, y2),
            ) {
                (Some(first), Some(second)) => (first.0[0], second.0[0]),
                (_, Some(second)) => (0, second.0[0]),
                (Some(first), _) => (first.0[0], 0),
                (_, _) => (0, 0),
            };

            brief_descriptor[i] += if first > second { 1 } else { 0 };
            brief_descriptor[i] <<= 1;
        }
    }

    brief_descriptor
}

#[derive(Default)]
pub struct KeyframeDatabase {
    key_frames: Vec<Frame>,
}
