use image::{imageops::grayscale_with_type, ImageBuffer, Pixel, Rgba};
use imageproc::drawing;
use nalgebra::Matrix3;
use once_cell::sync::Lazy;

use crate::{
    frame::{Feature, Frame, KeyFrame},
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
    /// Create the Local Mapping and Loop Closing
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track_monocular(
        &mut self,
        image_buffer: &mut ImageBuffer<Rgba<u8>, &mut [u8]>,
        timestamp: f64,
    ) {
        let grayscale_image = grayscale_with_type(image_buffer);
        let features: Vec<Feature> = Feature::from_fast_and_brief_128(&grayscale_image);

        if let Some(last_features) = &self.tracker.last_key_frame {
            for Feature { keypoint: kp, .. } in &last_features.frame.features {
                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    // draw the keypoint onto the image
                    (kp.x as _, kp.y as _),
                    // draw everything as a dot
                    1,
                    *GREEN,
                );
            }

            for Feature { keypoint: kp, .. } in &features {
                drawing::draw_hollow_circle_mut(
                    image_buffer,
                    // draw the keypoint onto the image
                    (kp.x as _, kp.y as _),
                    // draw everything as a dot
                    1,
                    *GREEN,
                );
            }

            // draw matches
            for (Feature { keypoint: kp1, .. }, Feature { keypoint: kp2, .. }) in
                compute_matches(&features, &last_features.frame.features)
            {
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
    return feature1.keypoint.x.abs_diff(feature2.keypoint.x).pow(2)
        + feature1.keypoint.y.abs_diff(feature2.keypoint.y).pow(2)
        < 50;
}

static GREEN: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 255, 0, 255]));
static BLUE: Lazy<Rgba<u8>> = Lazy::new(|| *Rgba::from_slice(&[0, 0, 255, 255]));
