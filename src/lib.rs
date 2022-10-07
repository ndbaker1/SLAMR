mod frame;
mod loop_closing;
mod system;
mod tracking;

#[cfg(test)]
mod tests {
    use crate::system::System;

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

        // Stop system
        slam.shutdown();
    }
}
