mod frame;
mod loop_closure;
mod system;
mod tracking;

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use image::{imageops::grayscale, GrayImage};

    use crate::system::System;

    #[test]
    fn on_system_execution() {
        let slam: System = todo!();

        //// process sensor 1 (imu)
        //let (imu_sender, imu_receiver) = sync::mpsc::channel::<u8>();
        //thread::spawn(move || loop {
        //    imu_sender.send(1u8);
        //});

        //// process sensor 2 (camera)
        //let (camera_sender, camera_receiver) = sync::mpsc::channel::<u8>();
        //thread::spawn(move || loop {
        //    camera_sender.send(2u8);
        //});

        loop {
            // imu_receiver.try_recv();
            // camera_receiver.try_recv();

            //slam.track_monocular(&frame, 0.0);

            thread::sleep(Duration::from_millis(10));

            break;
        }

        // Stop system
        slam.shutdown();
    }
}
