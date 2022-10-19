use image::{ImageBuffer, Rgba};
use nalgebra::Matrix3;

use crate::tracker::{ImuMeasurment, Tracker};

#[derive(Default)]
pub struct System {
    pub camera_intrinsics: (Matrix3<f64>, Matrix3<f64>),
    /// KeyFrame database for place recognition (relocalization and loop detection).
    //KeyFrameDatabase* mpKeyFrameDatabase;

    /// Tracker. It receives a frame and computes the associated camera pose.
    /// It also decides when to insert a new keyframe, create some new MapPoints and
    /// performs relocalization if tracking fails.
    pub tracker: Tracker,
}

// TODO - Local Bundle Adjustment & Global Bundle Adjustment

impl System {
    pub fn track_monocular_inertial(
        &mut self,
        image_buffer: Option<&mut ImageBuffer<Rgba<u8>, &mut [u8]>>,
        imu_measurements: &[ImuMeasurment],
        _timestamp: f64,
    ) {
        if let Some(image_buffer) = image_buffer {
            let rt = self.tracker.get_rt_pose_change(image_buffer);
        }
        self.tracker.process_imu_measurements(imu_measurements);
    }

    // Run any important cleanup procedures then SLAM instance is no longer in use
    pub fn shutdown(self) {
        // TODO: cleanup procedures
    }
}

type T = f64;
pub fn get_camera_intrinsic(f: T, w: T, h: T) -> Matrix3<T> {
    Matrix3::new(f, 0.0, w / 2.0, 0.0, f, h / 2.0, 0.0, 0.0, 1.0)
}
