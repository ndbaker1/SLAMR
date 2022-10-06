use std::sync::Mutex;

#[cfg(test)]
mod tests {
    use super::*;

    fn example_mono_execution() {
        let slam: System = todo!();
        // Main loop
        let im: _ = todo!();
        loop {
            // Read image from file
            im = todo!(); // cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
            let tframe = todo!(); // vTimestamps[ni];

            if true {
                //im.empty() {
                println!("no further frames");
                break;
            }

            // Pass the image to the SLAM system
            slam.track_monocular(im, tframe);
        }

        // Stop all threads
        slam.shutdown();
    }
}

pub enum InputSensor {
    Monocular = 1,
    Stereo = 2,
    Rgbd = 3,
}

struct MapPoint {}
struct KeyPoint {}

pub struct System {
    pub sensor: InputSensor,
    // ORB vocabulary used for place recognition and feature matching.
    //ORBVocabulary* mppubeVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    //KeyFrameDatabase* mpKeyFrameDatabase;

    //// Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;

    //// Tracker. It receives a frame and computes the associated camera pose.
    //// It also decides when to insert a new keyframe, create some new MapPoints and
    //// performs relocalization if tracking fails.
    //Tracking* mpTracker;

    //// Local Mapper. It manages the local map and performs local bundle adjustment.
    //LocalMapping* mpLocalMapper;

    //// Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    //// a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    //LoopClosing* mpLoopCloser;

    //// The viewer draws the map and the current camera pose. It uses Pangolin.
    //Viewer* mpViewer;

    //FrameDrawer* mpFrameDrawer;
    //MapDrawer* mpMapDrawer;

    //// System threads: Local Mapping, Loop Closing, Viewer.
    //// The Tracking thread "lives" in the main execution thread that creates the System object.
    //std::thread* mptLocalMapping:
    //std::thread* mptLoopClosing;
    //std::thread* mptViewer;

    //// Reset flag
    m_mutex_reset: Mutex<()>,
    mb_reset: bool,

    /// Change mode flags
    m_mutex_mode: Mutex<()>,
    mb_activate_localization_mode: bool,
    mb_deactivate_localization_mode: bool,

    /// Tracking state
    m_tracking_state: u32,
    m_tracked_map_points: Vec<MapPoint>,
    m_tracked_key_points_un: Vec<KeyPoint>,
    m_mutex_state: Mutex<()>,
}

impl System {
    pub fn track_monocular(&mut self, matrix: &[u8], timestamp: f64) {}
    fn activate_localization_mode(&mut self) {}
    fn deactivate_localization_mode(&mut self) {}
    fn map_changed(&self) -> bool {
        todo!()
    }
    fn reset(&mut self) {}
    pub fn shutdown(self) {}
}
