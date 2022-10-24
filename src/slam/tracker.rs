use std::collections::HashSet;

use arrsac::Arrsac;
use bitarray::BitArray;
use image::{imageops::grayscale_with_type, ImageBuffer, Pixel, Rgba};
use imageproc::drawing;
use nalgebra::{convert, Matrix3, RowVector3, SMatrix, Vector2, Vector3};
use once_cell::sync::Lazy;
use sample_consensus::{Consensus, Estimator, Model};
use space::{Knn, KnnFromBatch, LinearKnn, Metric};

use crate::{
    algorithms::{
        brief::{features_using_fast, BinaryDescriptor},
        camera::disambiguate_camera_pose,
        triangulation::triangulate_linear,
    },
    slam::features::Feature,
    slam::frame::Frame,
};

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
    /// The most recently seen frame
    pub current_frame: Frame<SizedFeature>,
    /// The previous frame
    pub last_frame: Option<Frame<SizedFeature>>,
    /// Camera Extrinsic Matrix represented by a `3x4` Matrix:
    /// ```plain
    /// [R -T] = R[I -T]
    /// R: 3D Rotation Matrix
    /// T: 3D Translation Matrix
    /// ```
    pub camera_extrinsic: Option<(Vector3<f64>, Matrix3<f64>)>,
    /// Camera Intrinsic `3x3` Matrix known as `K`,
    /// which is made up of a 2D Translation, 2D Scaling, and 2D Shear
    pub camera_intrinsic: Matrix3<f64>,
}

impl Tracker {
    /// Structure from Motion Pipeline used for mutliple perspectives from a single camera
    pub fn visual_structure_from_motion(
        &mut self,
        image_buffer: &mut ImageBuffer<Rgba<u8>, &mut [u8]>,
    ) {
        let grayscale_image = grayscale_with_type(image_buffer);
        let features: Vec<SizedFeature> = features_using_fast(&grayscale_image)
            .into_iter()
            .map(|(descriptor, keypoint)| SizedFeature {
                descriptor,
                keypoint,
            })
            .collect();

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

            let matches = match features.len() {
                1.. => {
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
                }
                _ => Vec::new(),
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
                for (m1, m2) in inliers.iter().map(|&i| matches[i]) {
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
                }

                let essential = fundamental.into_essential(&self.camera_intrinsic);
                let (c_set, r_set) = essential.extract_pose_configurations();

                macro_rules! triangulate_group {
                    ($index:expr) => {
                        triangulate_linear(
                            &self.camera_intrinsic,
                            &self.camera_extrinsic.unwrap_or_default().0,
                            &self.camera_extrinsic.unwrap_or_default().1,
                            &c_set[$index],
                            &r_set[$index],
                            &inliers
                                .iter()
                                .map(|&k| matches[k])
                                .map(|(feature, _)| &feature.keypoint)
                                .map(|kp| Vector2::new(kp.x as f64, kp.y as f64))
                                .collect::<Vec<_>>(),
                            &inliers
                                .iter()
                                .map(|&k| matches[k])
                                .map(|(_, feature)| &feature.keypoint)
                                .map(|kp| Vector2::new(kp.x as f64, kp.y as f64))
                                .collect::<Vec<_>>(),
                        )
                    };
                }

                let x_sets = [
                    triangulate_group!(0),
                    triangulate_group!(1),
                    triangulate_group!(2),
                    triangulate_group!(3),
                ];

                // final data for camera pose and 3D coordinates
                let (c, r, x_set) = disambiguate_camera_pose(&c_set, &r_set, &x_sets);

                // update stored pose matricies
                if self.camera_extrinsic.is_none() {
                    self.camera_extrinsic = Some((c, r));
                }

                for x in x_set {
                    drawing::draw_hollow_circle_mut(
                        image_buffer,
                        // changing the dimentions back to normal. WIP
                        ((x.x * 150.0) as _, (x.y * 150.0) as _),
                        2,
                        *PURPLE,
                    );
                }
            }
        }

        self.last_frame = Some(Frame {
            features,
            ..Default::default()
        })
    }

    ///
    pub fn process_imu_measurements(&mut self, imu_measurements: &[ImuMeasurment]) {
        //
    }
}

pub type Quaternion = [f64; 4];
pub type Vec3 = [f64; 3];

pub enum ImuMeasurment {
    OrientationQ(Quaternion),
    Acceleration(Vec3),
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

struct EssentialMatrix<T>(Matrix3<T>);

impl EssentialMatrix<f64> {
    /// Convert from Essential Matrix to Rt (Rotation and Translation)
    /// Reference: https://ia601408.us.archive.org/view_archive.php?archive=/7/items/DIKU-3DCV2/DIKU-3DCV2.zip&file=DIKU-3DCV2%2FHandouts%2FLecture16.pdf
    fn extract_pose_configurations(&self) -> ([Vector3<f64>; 4], [Matrix3<f64>; 4]) {
        let matrix_w = Matrix3::from_rows(&[
            RowVector3::new(0.0, -1.0, 0.0),
            RowVector3::new(1.0, 0.0, 0.0),
            RowVector3::new(0.0, 0.0, 1.0),
        ]);

        // U,d,Vt = np.linalg.svd(F)
        let svd = self.0.svd(true, true);
        let matrix_u = svd.u.unwrap();
        let matrix_v_t = svd.v_t.unwrap();

        // R = U W_inv V_T
        let rotation1 = matrix_u * matrix_w * matrix_v_t;
        let rotation2 = matrix_u * matrix_w.transpose() * matrix_v_t;

        // t = u_3 where: U[u_1, u_2, u_3]
        let camera_t = matrix_u.column(2).clone_owned();

        let sign1 = rotation1.determinant().signum();
        let sign2 = rotation2.determinant().signum();

        (
            [
                camera_t * sign1,
                -camera_t * sign1,
                camera_t * sign2,
                -camera_t * sign2,
            ],
            [
                rotation1 * sign1,
                rotation1 * sign1,
                rotation2 * sign2,
                rotation2 * sign2,
            ],
        )
    }
}

// Implementations for `sample_consensus`

struct FundamentalMatrix<T>(Matrix3<T>);

impl FundamentalMatrix<f64> {
    /// ### Assumptions
    /// 1. The camera intrinsic matrix for both point correspondences since it is the same camera.
    fn into_essential(self, camera_instrinsic: &Matrix3<f64>) -> EssentialMatrix<f64> {
        // E = K_T F K
        let essential = camera_instrinsic.try_inverse().unwrap() * self.0 * camera_instrinsic;

        // U, S, V_T = np.linalg.svd(E)
        let svd = essential.svd(true, true);
        let matrix_v_t = svd.v_t.unwrap();
        let matrix_u = svd.u.unwrap();

        // Create a Rank 2 matrix to eliminate noise
        let essential_rank2 = matrix_u * SMatrix::from_partial_diagonal(&[1.0, 1.0]) * matrix_v_t;

        EssentialMatrix(essential_rank2)
    }
}

type FeaturePair<'a> = (&'a SizedFeature, &'a SizedFeature);

impl<'a> Model<&'a FeaturePair<'a>> for FundamentalMatrix<f64> {
    /// Computes the distance from the Feature Keypoint
    /// to the epipolar line defined through the Funamental Matrix
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
    type Model = FundamentalMatrix<f64>;
    type ModelIter = std::iter::Once<FundamentalMatrix<f64>>;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = &'a FeaturePair<'a>> + Clone,
    {
        // Setup homogeneous linear equation as dst' * F * src = 0.

        const COLUMNS: usize = 9;

        // A = np.ones((src.shape[0], 9))
        let mut matrix_a = SMatrix::<f64, { Self::MIN_SAMPLES }, COLUMNS>::from_vec(
            [1f64; Self::MIN_SAMPLES * COLUMNS].to_vec(),
        );

        // A[:, :2] = src
        // A[:, :3] *= dst[:, 0, np.newaxis]
        // A[:, 3:5] = src
        // A[:, 3:6] *= dst[:, 1, np.newaxis]
        // A[:, 6:8] = src
        for (i, (src, dst)) in data.enumerate() {
            // normalize the points
            let x1: Vector2<f64> = convert(src.keypoint);
            let x2: Vector2<f64> = convert(dst.keypoint);

            let row = i * COLUMNS;
            matrix_a[row + 0] = x2.x * x1.x;
            matrix_a[row + 1] = x2.x * x1.y;
            matrix_a[row + 2] = x2.x;
            matrix_a[row + 3] = x2.y * x1.x;
            matrix_a[row + 4] = x2.y * x1.y;
            matrix_a[row + 5] = x2.y;
            matrix_a[row + 6] = x1.x;
            matrix_a[row + 7] = x1.y;
        }

        // Solve for the nullspace of the constraint matrix.

        // _, _, V_T = np.linalg.svd(A)
        let matrix_v_t = matrix_a
            .svd(false, true)
            .v_t
            .expect("could not solve transpose(V) in SVD of 8-point algorithm.");

        // F = V[-1, :].reshape(3, 3)
        // The nullsapce is equal to the last column of V, which is the bottom row in V_t
        // This is our Fundamental Matrix result
        let fundamental = Matrix3::from_row_iterator(matrix_v_t.row(7).iter().copied());

        std::iter::once(FundamentalMatrix(fundamental))
    }
}

//#[cfg(feature = "color")]
static RED: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[255, 0, 0, 255]));
//#[cfg(feature = "color")]
static BLUE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 0, 255, 255]));
//#[cfg(feature = "color")]
static GREEN: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 255, 0, 255]));
//#[cfg(feature = "color")]
static PURPLE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[255, 0, 255, 255]));
