use image::{ImageBuffer, Luma};
use imageproc::corners::{corners_fast9, Corner};
use nalgebra::Vector2;

use crate::algorithms;

pub type BinaryDescriptor<const N: usize> = [u8; N];

/// Feature object which holds a coordinate/pixel on a page
/// and tries to handle a generic descriptor
#[derive(Clone)]
pub struct Feature<Descriptor> {
    pub keypoint: Vector2<u32>,
    pub descriptor: Descriptor,
}

impl<const N: usize> Default for Feature<BinaryDescriptor<N>> {
    fn default() -> Self {
        Self {
            keypoint: Vector2::default(),
            descriptor: [0; N],
        }
    }
}

impl<A> Feature<A> {
    /// Uses FAST (Features from Accelerated Segment Test)
    /// as a keypoint detector for features like corners in a grayscale image
    fn fast_keypoints(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Vector2<u32>> {
        const FAST_CORNERS_THESHOLD: u8 = 35;

        corners_fast9(image, FAST_CORNERS_THESHOLD)
            .into_iter()
            .map(|Corner { x, y, .. }| Vector2::new(x, y))
            .collect()
    }
}

impl<const N: usize> Feature<BinaryDescriptor<N>> {
    /// applies the BRIEF (Binary Robust Independent Elementary Features) to compute descriptors for keypoints.
    pub fn from_fast_and_brief(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Self> {
        // using a kernel value of 2 indicated by reference:
        // https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6
        const GAUSSIAN_KERNEL_SIGMA: f32 = 2.0;

        // apply a guassion blur to the image for computing BRIEF descriptors,
        // that way the image is not overly sesnsitive to high frequency noise.
        let smoothed_image = imageproc::filter::gaussian_blur_f32(&image, GAUSSIAN_KERNEL_SIGMA);

        Self::fast_keypoints(image)
            .into_iter()
            .map(|keypoint| Feature {
                descriptor: algorithms::brief::compute_descriptor(
                    keypoint.x,
                    keypoint.y,
                    &smoothed_image,
                ),
                keypoint,
            })
            .collect()
    }
}
