use crate::{
    frame::{Frame, KeyFrame},
    system::InputSensor,
};

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
pub struct Tracking {
    pub tracking_state: TrackingState,
    pub last_tracking_state: TrackingState,

    pub sensor: InputSensor,

    pub current_frame: Frame,
    pub last_frame: Frame,

    pub initial_frame: Frame,

    last_key_frame: Option<KeyFrame>,
}

impl Tracking {
    pub fn new() -> Self {
        Self::default()
    }

    /// main tracking function
    pub fn track(&mut self) {
        if matches!(self.tracking_state, TrackingState::NoImagesYet) {
            if self.current_frame.timestamp > self.last_frame.timestamp + 1.0 {
                //
            }
        }

        if let (
            InputSensor::IMUMonocular | InputSensor::IMUStereo | InputSensor::IMURgbd,
            Some(last_key_frame),
        ) = (&self.sensor, &self.last_key_frame)
        {
            self.current_frame.imu_bias = last_key_frame.imu_bias.clone();
        }

        if matches!(
            self.tracking_state,
            TrackingState::Ok | TrackingState::RecentlyLost
        ) {
            if self.current_frame.is_set() {
                // Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF.GetPoseInverse();
                // mlRelativeFramePoses.push_back(Tcr_);
                // mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                // mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                // mlbLost.push_back(mState==LOST);
            } else {
                // This can happen if tracking is lost
                // mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                // mlpReferences.push_back(mlpReferences.back());
                // mlFrameTimes.push_back(mlFrameTimes.back());
                // mlbLost.push_back(mState == LOST);
            }
        }
    }

    pub fn reset() {}
    pub fn reset_active_map() {}

    fn preintegrate_imu() {}
    fn reset_frame_imu() {}
}
