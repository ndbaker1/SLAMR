use image::{imageops::grayscale_with_type, ImageBuffer, Pixel, Rgba};
use imageproc::{
    corners::{corners_fast9, Corner},
    drawing,
};
use nalgebra::Matrix3;
use once_cell::sync::Lazy;

use crate::{
    frame::{Descriptor, Feature, Frame, KeyFrame, KeyPoint},
    tracking::Tracking,
};

struct MapPoint {}

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
    /// Create the Local Mapping and Loop Closing
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track_monocular(
        &mut self,
        image_buffer: &mut ImageBuffer<Rgba<u8>, &mut [u8]>,
        timestamp: f64,
    ) {
        // first step - detect viable feature keypoints using the FAST-detection algorithm
        // second step - compute descriptors about keypoints (using BRIEF?)
        let features: Vec<_> = corners_fast9(&grayscale_with_type(image_buffer), 20)
            .into_iter()
            .map(|Corner { x, y, .. }| {
                let keypoint = KeyPoint { x, y };
                let descriptor = compute_descriptor(&keypoint);
                (keypoint, descriptor)
            })
            .collect();

        if let Some(last_features) = &self.tracker.last_key_frame {
            for (keypoint, _) in &last_features.frame.features {
                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    // draw the keypoint onto the image
                    (keypoint.x as _, keypoint.y as _),
                    // draw everything as a dot
                    1,
                    *GREEN,
                );
            }

            for (keypoint, _) in &features {
                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    // draw the keypoint onto the image
                    (keypoint.x as _, keypoint.y as _),
                    // draw everything as a dot
                    1,
                    *GREEN,
                );
            }

            // draw matches
            for ((kp1, _), (kp2, _)) in compute_matches(&features, &last_features.frame.features) {
                drawing::draw_line_segment_mut(
                    image_buffer,
                    (kp1.x as _, kp1.y as _),
                    (kp2.x as _, kp2.y as _),
                    *BLUE,
                );
            }
        }

        self.tracker.last_key_frame = Some(KeyFrame {
            frame: Frame {
                features,
                ..Default::default()
            },
        });
    }

    pub fn shutdown(self) {}
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

fn compute_descriptor(keypoint: &KeyPoint) -> Descriptor {
    return 5;
}

fn compute_matches<'a>(
    features1: &'a Vec<Feature>,
    features2: &'a Vec<Feature>,
) -> Vec<(&'a Feature, &'a Feature)> {
    let mut matches = Vec::new();
    for feature1 in features1 {
        for feature2 in features2 {
            if features_compatible(feature1, feature2) {
                matches.push((feature1, feature2));
            }
        }
    }
    return matches;
}

fn features_compatible(feature1: &Feature, feature2: &Feature) -> bool {
    return feature1.0.x.abs_diff(feature2.0.x).pow(2) + feature1.0.y.abs_diff(feature2.0.y).pow(2)
        < 50;
}

static GREEN: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 255, 0, 255]));
static BLUE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 0, 255, 255]));
