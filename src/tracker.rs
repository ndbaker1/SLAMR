use std::collections::HashSet;

use arrsac::Arrsac;
use bitarray::BitArray;
use image::{imageops::grayscale_with_type, ImageBuffer, Pixel, Rgba};
use imageproc::drawing;
use nalgebra::{Matrix3, Matrix4, SMatrix, Vector3};
use once_cell::sync::Lazy;
use sample_consensus::{Consensus, Estimator, Model};
use space::{Knn, KnnFromBatch, LinearKnn, Metric};

use crate::frame::{BinaryDescriptor, Feature, Frame};

/// Const Generic assignement of Feature Descriptor Size
const DESCRIPTOR_SIZE: usize = 512 / u8::BITS as usize;
type SizedFeature = Feature<BinaryDescriptor<DESCRIPTOR_SIZE>>;

/// Tracking states
#[derive(Default)]
pub enum TrackingState {
    #[default]
    SystemNotReady = -1,
    NoImagesYet = 0,
    NotInitialized = 1,
    Ok = 2,
    RecentlyLost = 3,
    Lost = 4,
    OkKlt = 5,
}

#[derive(Default)]
pub struct Tracker {
    pub tracking_state: TrackingState,
    pub last_tracking_state: TrackingState,
    pub current_frame: Frame<SizedFeature>,
    pub last_frame: Option<Frame<SizedFeature>>,
}

impl Tracker {
    /// Using the Image and Visual Odometry find Rt: the Rotation and Translation matricies.
    pub fn get_rt_pose_change(&mut self, image_buffer: &mut ImageBuffer<Rgba<u8>, &mut [u8]>) {
        let grayscale_image = grayscale_with_type(image_buffer);
        let features: Vec<SizedFeature> = Feature::from_fast_and_brief(&grayscale_image);

        if let Some(last_frame) = &self.last_frame {
            // draw raw features
            #[cfg(feature = "featues")]
            {
                for Feature { keypoint, .. } in &last_frame.features {
                    drawing::draw_hollow_circle_mut(
                        image_buffer,
                        // draw the keypoint onto the image
                        (keypoint.x as _, keypoint.y as _),
                        // draw everything as a dot
                        1,
                        *RED,
                    );
                }

                for Feature { keypoint, .. } in &features {
                    drawing::draw_hollow_circle_mut(
                        image_buffer,
                        // draw the keypoint onto the image
                        (keypoint.x as _, keypoint.y as _),
                        // draw everything as a dot
                        1,
                        *GREEN,
                    );
                }
            }

            // Prepare feature data to perform kNN matching
            // TODO: I dont like this requirement on owned memory. find a different implementation or write one.
            let data = features.iter().map(|f| (f, 1u8)).collect::<Vec<_>>();
            let search: LinearKnn<FeatureHamming, _> = KnnFromBatch::from_batch(data.iter());

            let matches = if features.len() > 1 {
                // enforce that each point maps to only one
                let mut seen_current = HashSet::<usize>::new();
                let mut seen_last = HashSet::<usize>::new();

                // Find the mappings from the last frame onto the current frame using kNN (K Nearest Neighbor).
                last_frame
                    .features
                    .iter()
                    .enumerate()
                    .filter_map(|(i, feature)| {
                        // find k = 2 nearest neighbors and then perform Lowe's test to filter out answers potentiall chosen by noise
                        // https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
                        let nearest = search.knn(&feature, 2);

                        // this value is kind of high, but it wont give that many points if i make it any lower ಥ_ಥ
                        const LOWE_RATIO: f32 = 0.75;
                        // scale the threshold with the size of the descriptor?
                        const DISTANCE_THRESHOLD: u32 = DESCRIPTOR_SIZE as _;

                        // filter the results to have tolerable hamming distances
                        if nearest.iter().any(|q| q.0.distance < DISTANCE_THRESHOLD)
                        // then apply Lowe's ratio test 
                        && nearest[0].0.distance < (LOWE_RATIO * nearest[1].0.distance as f32) as u32
                        // check that this point has not already been assigned a correspondence
                        && !seen_current.contains(&nearest[0].0.index)
                        && !seen_last.contains(&i)
                        {
                            // record correspondence
                            seen_current.insert(nearest[0].0.index);
                            seen_last.insert(i);

                            Some((feature, *nearest[0].1))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            #[cfg(feature = "matched")]
            for &(m1, m2) in &matches {
                // draw matches versions of features
                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    (m1.keypoint.x as _, m1.keypoint.y as _),
                    1,
                    *RED,
                );

                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    (m2.keypoint.x as _, m2.keypoint.y as _),
                    1,
                    *GREEN,
                );

                // draw line connecting past to present features
                drawing::draw_line_segment_mut(
                    image_buffer,
                    (m1.keypoint.x as _, m1.keypoint.y as _),
                    (m2.keypoint.x as _, m2.keypoint.y as _),
                    *BLUE,
                );
            }

            // Perform Random Sampling Consensus technique to find the Essential Matrix.
            // Apply the 8-point algorithm to estimate the Fundamental Matrix, then use this estimated value for
            // Sampling Consensus in order to eliminate outlier points and isolate the most correct Essential Matrix.
            // The Essential Matrix will contain the Rotation and Translation from the previous frame to the next.
            const INLIER_THRESHOLD: f64 = 1.8;

            // This is an ARRSAC (Adaptive Real-Time Random Sample Consensus) package,
            // which could perform as well or better than RANSAC.
            // https://people.inf.ethz.ch/pomarc/pubs/RaguramECCV08.pdf

            if let Some((fundamental, inliers)) = Arrsac::new(INLIER_THRESHOLD, rand::thread_rng())
                .model_inliers(&EssentialMatrixEstimator, matches.iter())
            {
                // draw matches versions of features
                // remove the outliers from the matched data set
                // #[cfg(feature = "inliers")]
                for (m1, m2) in inliers.into_iter().map(|i| matches[i]) {
                    drawing::draw_hollow_circle_mut(
                        image_buffer,
                        (m1.keypoint.x as _, m1.keypoint.y as _),
                        1,
                        *RED,
                    );

                    drawing::draw_hollow_circle_mut(
                        image_buffer,
                        (m2.keypoint.x as _, m2.keypoint.y as _),
                        1,
                        *GREEN,
                    );

                    drawing::draw_line_segment_mut(
                        image_buffer,
                        (m1.keypoint.x as _, m1.keypoint.y as _),
                        (m2.keypoint.x as _, m2.keypoint.y as _),
                        *BLUE,
                    );

                    // Convert from Essential Matrix to Rt

                    // W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
                    let matrix_w =
                        Matrix3::<f64>::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

                    // U,d,Vt = np.linalg.svd(F)
                    let svd = matrix_w.svd(true, true);
                    let mut matrix_u = svd.u.unwrap();
                    let mut matrix_v = svd.v_t.unwrap();

                    // if np.linalg.det(U) < 0:
                    //     U *= -1.0
                    if matrix_u.determinant() < 0.0 {
                        matrix_u *= -1.0;
                    }

                    // if np.linalg.det(Vt) < 0:
                    //     Vt *= -1.0
                    if matrix_v.determinant() < 0.0 {
                        matrix_v *= -1.0;
                    }

                    // R = np.dot(np.dot(U, W), Vt)
                    // if np.sum(R.diagonal()) < 0:
                    //     R = np.dot(np.dot(U, W.T), Vt)
                    let mut matrix_r = matrix_v * matrix_u.dot(&matrix_w);
                    if matrix_r.diagonal().sum() < 0.0 {
                        matrix_r = matrix_v * matrix_u.dot(&matrix_w.transpose())
                    }

                    // t = U[:, 2]
                    // # TODO: Resolve ambiguities in better ways. This is wrong.
                    // if t[2] < 0:
                    //     t *= -1
                    let mut matrix_t = matrix_u.column_mut(2);
                    if matrix_t[2] < 0.0 {
                        matrix_t *= -1.0;
                    }

                    // return np.linalg.inv(poseRt(R, t))
                    // ret = np.eye(4)
                    // ret[:3, :3] = R
                    // ret[:3, 3] = t
                    let mut matrix_final = Matrix4::<f64>::identity();
                    for y in 0..3 {
                        for x in 0..3 {
                            matrix_final[x * 3 + y] = matrix_r[x * 3 + y];
                        }
                        matrix_final[y * 3 + 3] = matrix_t[y];
                    }

                    // TODO: should be invertable.
                    // matrix_final.try_inverse().unwrap();
                }
            }
        }

        self.last_frame = Some(Frame {
            features,
            ..Default::default()
        });
    }

    pub fn process_imu_measurements(&mut self, imu_measurements: &[ImuMeasurment]) {
        //
    }
}

// Implementations for `space`

#[derive(Default)]
struct FeatureHamming;

impl<'f> Metric<&'f SizedFeature> for FeatureHamming {
    type Unit = u32;
    fn distance(&self, a: &&SizedFeature, b: &&SizedFeature) -> Self::Unit {
        BitArray::new(a.descriptor).distance(&BitArray::new(b.descriptor))
    }
}

pub type Quaternion = [f64; 4];
pub type Vec3 = [f64; 3];

pub enum ImuMeasurment {
    OrientationQ(Quaternion),
    Acceleration(Vec3),
}

// Implementations for `sample_consensus`

pub struct FundamentalMatrix(Matrix3<f64>);

type FeaturePair<'a> = (&'a SizedFeature, &'a SizedFeature);

impl<'a> Model<&'a FeaturePair<'a>> for FundamentalMatrix {
    fn residual(&self, data: &&FeaturePair<'a>) -> f64 {
        let src_homogenous = Vector3::new(data.0.keypoint.x as _, data.0.keypoint.y as _, 1f64);
        let dst_homogenous = Vector3::new(data.1.keypoint.x as _, data.1.keypoint.y as _, 1f64);
        let src_f = src_homogenous.transpose() * self.0;
        let dst_f = dst_homogenous.transpose() * self.0;
        let src_f_distance = (src_f * dst_homogenous).sum().abs();

        src_f_distance
            / f64::sqrt(
                src_f.x * src_f.x + src_f.y * src_f.y + dst_f.x * dst_f.x + dst_f.y * dst_f.y,
            )
    }
}

#[derive(Default)]
struct EssentialMatrixEstimator;

impl<'a> Estimator<&'a FeaturePair<'a>> for EssentialMatrixEstimator {
    const MIN_SAMPLES: usize = 8;
    type Model = FundamentalMatrix;
    type ModelIter = std::iter::Once<FundamentalMatrix>;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = &'a FeaturePair<'a>> + Clone,
    {
        // Setup homogeneous linear equation as dst' * F * src = 0.

        const ROWS: usize = 8;
        const COLUMNS: usize = 9;

        // A = np.ones((src.shape[0], 9))
        let mut matrix_a = SMatrix::<f64, ROWS, COLUMNS>::from_vec([1f64; ROWS * COLUMNS].to_vec());

        // A[:, :2] = src
        // A[:, :3] *= dst[:, 0, np.newaxis]
        // A[:, 3:5] = src
        // A[:, 3:6] *= dst[:, 1, np.newaxis]
        // A[:, 6:8] = src
        for (i, (src, dst)) in data.enumerate() {
            matrix_a[i * COLUMNS + 0] = (dst.keypoint.x * src.keypoint.x) as _;
            matrix_a[i * COLUMNS + 1] = (dst.keypoint.x * src.keypoint.y) as _;
            matrix_a[i * COLUMNS + 2] = dst.keypoint.x as _;
            matrix_a[i * COLUMNS + 3] = (dst.keypoint.y * src.keypoint.x) as _;
            matrix_a[i * COLUMNS + 4] = (dst.keypoint.y * src.keypoint.y) as _;
            matrix_a[i * COLUMNS + 5] = dst.keypoint.y as _;
            matrix_a[i * COLUMNS + 6] = src.keypoint.x as _;
            matrix_a[i * COLUMNS + 7] = src.keypoint.y as _;
        }

        // Solve for the nullspace of the constraint matrix.

        // _, _, V = np.linalg.svd(A)
        let matrix_v = matrix_a.svd(false, true).v_t.unwrap();

        // F = V[-1, :].reshape(3, 3)
        let matrix_f = Matrix3::from_row_iterator(matrix_v.row(7).iter().copied());

        // Enforcing the internal constraint that two singular values must be non-zero and one must be zero.
        // Create a Rank 2 matrix.

        // U, S, V = np.linalg.svd(F)
        let svd = matrix_f.svd(true, true);
        let mut matrix_s = svd.singular_values;
        let matrix_v = svd.v_t.unwrap();
        let matrix_u = svd.u.unwrap();

        // S[0] = S[1] = (S[0] + S[1]) / 2.0
        // S[2] = 0
        // self.params = U @ np.diag(S) @ V
        let mean = (matrix_s[0] + matrix_s[1]) / 2.0;
        matrix_s[0] = mean;
        matrix_s[1] = mean;
        matrix_s[2] = 0.0;
        let rank_2 =
            matrix_u * SMatrix::from_partial_diagonal(&[matrix_s[0], matrix_s[1]]) * matrix_v;

        std::iter::once(FundamentalMatrix(rank_2))
    }
}

//#[cfg(feature = "color")]
static RED: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[255, 0, 0, 255]));
//#[cfg(feature = "color")]
static BLUE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 0, 255, 255]));
//#[cfg(feature = "color")]
static GREEN: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 255, 0, 255]));
