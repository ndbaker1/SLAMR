use crate::frame::{Frame, KeyFrame};

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

    pub current_frame: Frame,
    pub last_frame: Frame,

    pub initial_frame: Frame,

    pub last_key_frame: Option<KeyFrame>,
}
