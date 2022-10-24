use image::{GenericImageView, ImageBuffer, Luma};
use imageproc::corners::{corners_fast9, Corner};
use nalgebra::Vector2;
use once_cell::sync::Lazy;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Feature Descriptor that uses a byte-array representation
pub type BinaryDescriptor<const N: usize> = [u8; N];

/// applies the BRIEF (Binary Robust Independent Elementary Features) to compute descriptors for keypoints.
pub fn features_using_fast<const N: usize>(
    image: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> Vec<(BinaryDescriptor<N>, Vector2<u32>)> {
    // using a kernel value of 2 indicated by reference:
    // https://medium.com/data-breach/introduction-to-brief-binary-robust-independent-elementary-features-436f4a31a0e6
    const GAUSSIAN_KERNEL_SIGMA: f32 = 2.0;

    // apply a guassion blur to the image for computing BRIEF descriptors,
    // that way the image is not overly sesnsitive to high frequency noise.
    let smoothed_image = imageproc::filter::gaussian_blur_f32(image, GAUSSIAN_KERNEL_SIGMA);

    const FAST_CORNERS_THESHOLD: u8 = 35;

    corners_fast9(image, FAST_CORNERS_THESHOLD)
        .into_iter()
        .map(|Corner { x, y, .. }| {
            (
                compute_descriptor(x, y, &smoothed_image),
                Vector2::new(x, y),
            )
        })
        .collect()
}

/// Compute BRIEF (Binary Robust Independent Elementary Features) on a given grayscale image given the target keypoint.
///
/// ### CAUTION
/// const N Generic should be less than `512 / u8::BITS = 64`
pub fn compute_descriptor<const N: usize>(
    x: u32,
    y: u32,
    image: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> BinaryDescriptor<N> {
    /// Precomputed samples taked un to 512 bits for BREIF point samples.
    /// The values remain consistent accross frames, because we want to achieve a similar level of entropy
    /// to best match our previous encounters with points.
    static BRIEF512_SAMPLES: Lazy<[[i16; 4]; 512]> = Lazy::new(|| {
        // use reproducible random numbers so that
        let mut rng = StdRng::seed_from_u64(42);

        // assuming that normal distribution sampling is going to be bounded by (+-) 10 for now.
        // this results in patches that are 20 x 20 pixels
        let normal_dist: Normal<f64> = Normal::new(0 as _, 2 as _).unwrap();

        let mut samples = [[0; 4]; 512];
        for sample in &mut samples {
            *sample = [
                normal_dist.sample(&mut rng) as _,
                normal_dist.sample(&mut rng) as _,
                normal_dist.sample(&mut rng) as _,
                normal_dist.sample(&mut rng) as _,
            ];
        }

        samples
    });

    const BITS: usize = u8::BITS as _;

    let mut brief_descriptor = [0; N];
    for i in 0..N {
        for j in 0..BITS as usize {
            let [p1x, p1y, p2x, p2y] = BRIEF512_SAMPLES[i * BITS + j];

            let (x1, y1) = (x as i16 + p1x, y as i16 + p1y);
            let (x2, y2) = (x as i16 + p2x, y as i16 + p2y);
            // UNSAFETY JUSTIFICATION
            //  Correctness
            //      the range of (x,y) coordinate pairs will always
            //      have bounds manually checked before calling `unsafe_get_pixel(_,_)`.
            //  Fallback
            //      When the bounds are not satisfied for the function,
            //      we will always use 0 as the descriptor of the pixel in question.
            //      Remember that these are Grayscale images, where pixel values are `u8`

            let first =
                if x1 < image.width() as i16 && x1 >= 0 && y1 < image.height() as i16 && y1 >= 0 {
                    // SEE UNSAFETY JUSTIFICATION ABOVE
                    unsafe { image.unsafe_get_pixel(x1 as u32, y1 as u32).0[0] }
                } else {
                    0
                };

            let second =
                if x2 < image.width() as i16 && x2 >= 0 && y2 < image.height() as i16 && y2 >= 0 {
                    // SEE UNSAFETY JUSTIFICATION ABOVE
                    unsafe { image.unsafe_get_pixel(x2 as u32, y2 as u32).0[0] }
                } else {
                    0
                };

            brief_descriptor[i] += if first > second { 1 } else { 0 };
            brief_descriptor[i] <<= 1;
        }
    }

    brief_descriptor
}
