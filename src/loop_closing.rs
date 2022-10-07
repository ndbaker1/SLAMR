use std::{collections::VecDeque, thread::sleep, time::Duration};

use crate::frame::KeyFrame;

#[derive(Default)]
pub struct LoopClosing {
    finished: bool,
    loop_key_frame_queue: VecDeque<KeyFrame>,
}

impl LoopClosing {
    pub fn run(&mut self) {
        self.finished = false;

        loop {
            if self.check_new_key_frames() {}

            if self.finished {
                break;
            }

            sleep(Duration::from_millis(5000))
        }
    }

    fn check_new_key_frames(&self) -> bool {
        self.loop_key_frame_queue.is_empty()
    }
}
