use std::{collections::VecDeque, thread::sleep, time::Duration};

use crate::frame::KeyFrame;

/// Loop Closure is used to reduce sensor drift accumulation
/// and improve the consistency of localization
///
/// By using a sliding window of KeyFrames (distinct Frames selected to represent the path)
#[derive(Default)]
pub struct LoopClosure {
    loop_keyframe_queue: VecDeque<KeyFrame>,
}

impl LoopClosure {
    pub fn run(&mut self) {
        loop {
            if let Some(last_keyframe) = self.loop_keyframe_queue.front() {
                if self.check_new_key_frames() {}
            }

            sleep(Duration::from_millis(500))
        }
    }

    fn check_new_key_frames(&self) -> bool {
        self.loop_keyframe_queue.is_empty()
    }
}
