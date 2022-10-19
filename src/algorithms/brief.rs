use image::{GenericImageView, ImageBuffer, Luma};
use once_cell::sync::Lazy;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Compute BRIEF (Binary Robust Independent Elementary Features) on a given grayscale image given the target keypoint.
///
/// ### CAUTION
/// const N Generic should be less than `512 / u8::BITS = 64`
pub fn compute_descriptor<const N: usize>(
    x: u32,
    y: u32,
    image: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> [u8; N] {
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
    for i in 0..512 {
        samples[i] = [
            normal_dist.sample(&mut rng) as _,
            normal_dist.sample(&mut rng) as _,
            normal_dist.sample(&mut rng) as _,
            normal_dist.sample(&mut rng) as _,
        ];
    }

    samples
});
