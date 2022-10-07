use crate::tracking::Tracking;

#[derive(Default)]
pub enum InputSensor {
    #[default]
    Monocular = 1,
    Stereo = 2,
    Rgbd = 3,
    IMUMonocular = 4,
    IMUStereo = 5,
    IMURgbd = 6,
}

struct MapPoint {}
struct KeyPoint {}

#[derive(Default)]
pub struct System {
    pub sensor: InputSensor,

    /// ORB vocabulary used for place recognition and feature matching.
    //ORBVocabulary* mppubeVocabulary;

    /// KeyFrame database for place recognition (relocalization and loop detection).
    //KeyFrameDatabase* mpKeyFrameDatabase;

    /// Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;

    /// Tracker. It receives a frame and computes the associated camera pose.
    /// It also decides when to insert a new keyframe, create some new MapPoints and
    /// performs relocalization if tracking fails.
    mp_tracker: Tracking,

    /// Local Mapper. It manages the local map and performs local bundle adjustment.
    //LocalMapping* mpLocalMapper;

    /// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    /// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    //LoopClosing* mpLoopCloser;

    /// System threads: Local Mapping, Loop Closing, Viewer.
    /// The Tracking thread "lives" in the main execution thread that creates the System object.
    //std::thread* mptLocalMapping:
    //std::thread* mptLoopClosing;
    //std::thread* mptViewer;

    /// Reset flag
    //m_mutex_reset: Mutex<()>,
    //mb_reset: bool,

    /// Change mode flags
    //m_mutex_mode: Mutex<()>,
    //mb_activate_localization_mode: bool,
    //mb_deactivate_localization_mode: bool,

    /// Tracking state
    m_tracking_state: u32,
    m_tracked_map_points: Vec<MapPoint>,
    m_tracked_key_points_un: Vec<KeyPoint>,
    //m_mutex_state: Mutex<()>,
}

impl System {
    /// Create the Local Mapping and Loop Closing
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track_monocular(&mut self, matrix: &[u8], timestamp: f64) {}
    fn activate_localization_mode(&mut self) {}
    fn deactivate_localization_mode(&mut self) {}
    fn map_changed(&self) -> bool {
        todo!()
    }
    fn reset(&mut self) {}
    pub fn shutdown(self) {}
}
