use arrsac::Arrsac;
use bitarray::BitArray;
use image::{imageops::grayscale_with_type, ImageBuffer, Pixel, Rgba};
use imageproc::drawing;
use nalgebra::Matrix3;
use once_cell::sync::Lazy;
use sample_consensus::{Consensus, Estimator};
use space::{Knn, KnnFromBatch, LinearKnn, Metric};

use crate::{
    frame::{Feature, Frame},
    tracking::Tracking,
};

#[derive(Default)]
pub struct System {
    pub camera_intrinsics: (Matrix3<f64>, Matrix3<f64>),
    /// ORB vocabulary used for place recognition and feature matching using Visual Bag of Words (VBoW)
    //ORBVocabulary* mppubeVocabulary;

    /// KeyFrame database for place recognition (relocalization and loop detection).
    //KeyFrameDatabase* mpKeyFrameDatabase;

    /// Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;

    /// Tracker. It receives a frame and computes the associated camera pose.
    /// It also decides when to insert a new keyframe, create some new MapPoints and
    /// performs relocalization if tracking fails.
    pub tracker: Tracking,
}

// TODO - Local Bundle Adjustment & Global Bundle Adjustment

impl System {
    pub fn track_monocular(
        &mut self,
        image_buffer: &mut ImageBuffer<Rgba<u8>, &mut [u8]>,
        timestamp: f64,
    ) {
        let grayscale_image = grayscale_with_type(image_buffer);
        let features: Vec<Feature> = Feature::from_fast_and_brief_128(&grayscale_image);

        if let Some(last_frame) = &self.tracker.last_key_frame {
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

            // prepare a data structure to perform knn
            // TODO: I dont like this requirement on owned memory. find a different implementation or write one.
            let data: Vec<_> = features.iter().map(|f| (f.clone(), 1)).collect();
            let search: LinearKnn<FeatureHamming, _> = KnnFromBatch::from_batch(data.iter());

            // Find the mappings from the last frame onto the current frame using kNN (K Nearest Neighbor).
            let matches = last_frame.features.iter().flat_map(|feature| {
                // uinsg a knn of 1 to just get the closest matched
                search
                    .knn(&feature, 1)
                    .into_iter()
                    .filter(|q| q.0.distance < 32)
                    .map(|(_, matched_feature, _)| (feature, matched_feature))
                    .collect::<Vec<_>>()
            });

            //  draw matches to the image for debugging
            for (m1, m2) in matches {
                drawing::draw_line_segment_mut(
                    image_buffer,
                    (m1.keypoint.x as _, m1.keypoint.y as _),
                    (m2.keypoint.x as _, m2.keypoint.y as _),
                    *BLUE,
                );
            }

            // if let Some(inliers) =
            //     Arrsac::new(20.0, rand::thread_rng()).model_inliers(todo!(), matches)
            // {
            //     //
            // }
        }

        self.tracker.last_key_frame = Some(Frame {
            features,
            ..Default::default()
        });
    }

    // Run any important cleanup procedures then SLAM instance is no longer in use
    pub fn shutdown(self) {
        // TODO: cleanup procedures
    }
}

type T = f64;
pub fn get_camera_intrinsic(f: T, w: T, h: T) -> Matrix3<T> {
    Matrix3::new(
        f,
        0 as _,
        w / 2.0,
        0 as _,
        f,
        h / 2.0,
        0 as _,
        0 as _,
        1 as _,
    )
}

#[derive(Default)]
struct FeatureHamming;

impl Metric<Feature> for FeatureHamming {
    type Unit = u32;
    fn distance(&self, a: &Feature, b: &Feature) -> Self::Unit {
        BitArray::new(a.descriptor).distance(&BitArray::new(b.descriptor))
    }
}

static GREEN: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 255, 0, 255]));
static RED: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[255, 0, 0, 255]));
static BLUE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 0, 255, 255]));
