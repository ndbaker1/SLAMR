use image::{ImageBuffer, Luma};
use imageproc::corners::{corners_fast9, Corner};
use once_cell::sync::Lazy;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Representation of a pixel point on an image
#[derive(Clone, Default)]
pub struct Keypoint {
    pub x: u32,
    pub y: u32,
}

pub type BinaryDescriptor<const N: usize> = [u8; N];

/// Feature object which holds a coordinate/pixel on a page
/// and tries to handle a generic descriptor
#[derive(Clone)]
pub struct Feature<Descriptor> {
    pub keypoint: Keypoint,
    pub descriptor: Descriptor,
}

impl<const N: usize> Default for Feature<BinaryDescriptor<N>> {
    fn default() -> Self {
        Self {
            keypoint: Keypoint::default(),
            descriptor: [0; N],
        }
    }
}

impl<A> Feature<A> {
    /// Uses FAST (Features from Accelerated Segment Test) as a keypoint detector for features like corners in a grayscale image,
    fn fast_keypoints(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Vec<Keypoint> {
        const FAST_CORNERS_THESHOLD: u8 = 35;

        corners_fast9(image, FAST_CORNERS_THESHOLD)
            .into_iter()
            .map(|Corner { x, y, .. }| Keypoint { x, y })
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
                descriptor: compute_brief(&keypoint, &smoothed_image),
                keypoint,
            })
            .collect()
    }
}

#[derive(Default)]
pub struct Frame<Feat> {
    pub features: Vec<Feat>,
    pub timestamp: f64,
}

/// Compute BRIEF (Binary Robust Independent Elementary Features) on a given grayscale image given the target keypoint.
///
/// ### CAUTION
/// const N Generic should be less than `512 / u8::BITS = 64`
fn compute_brief<const N: usize>(
    keypoint: &Keypoint,
    image: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> [u8; N] {
    const BITS: usize = u8::BITS as _;

    let mut brief_descriptor = [0; N];
    for i in 0..N {
        for j in 0..BITS as usize {
            let [p1x, p1y, p2x, p2y] = BRIEF512_SAMPLES[i * BITS + j];

            let (x1, y1) = (keypoint.x + p1x, keypoint.y + p1y);
            let (x2, y2) = (keypoint.x + p2x, keypoint.y + p2y);

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

/// Precomputed samples taked un to 512 bits for BREIF point samples.
/// The values remain consistent accross frames, because we want to achieve a similar level of entropy
/// to best match our previous encounters with points.
static BRIEF512_SAMPLES: Lazy<[[u32; 4]; 512]> = Lazy::new(|| {
    // use reproducible random numbers so that
    let mut rng = StdRng::seed_from_u64(42);

    // assuming that normal distribution sampling is going to be bounded by (+-) 10 for now.
    // this results in patches that are 20 x 20 pixels
    let normal_dist: Normal<f64> = Normal::new(0 as _, 2 as _).unwrap();

    let mut samples = [[0; 4]; 512];
    for i in 0..512 {
        samples[i] = [
            normal_dist.sample(&mut rng) as u32,
            normal_dist.sample(&mut rng) as u32,
            normal_dist.sample(&mut rng) as u32,
            normal_dist.sample(&mut rng) as u32,
        ];
    }

    samples
});

#[derive(Default)]
pub struct KeyframeDatabase<F> {
    key_frames: Vec<Frame<F>>,
}
