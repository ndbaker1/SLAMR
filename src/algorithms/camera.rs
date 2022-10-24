use nalgebra::{Matrix3, Vector3};

/// Correct Pose Data (Rotation and Translation) with the corresponding 3D-point
type CorrectPoseWithEstimation<'a> = (Vector3<f64>, Matrix3<f64>, &'a Vec<Vector3<f64>>);

const NUM_CONFIGURATIONS: usize = 4;

/// Determines which Camera Pose is correct by evaluating the [Cheirality Condition](http://users.cecs.anu.edu.au/~hartley/Papers/cheiral/revision/cheiral.pdf),
/// which is defined as when the transformed point lies in front of the camera.
///
/// Verify that the initial point has a Z-value greater than `0`, i.e. in front of the camera,
/// and then the same for the translated and rotated point (only about the Z-axis once again)
///
/// `X_3 > 0` (first camera) and `r_3(X − C) > 0` (transformation to second camera)
pub fn disambiguate_camera_pose<'a>(
    c_set: &[Vector3<f64>; NUM_CONFIGURATIONS],
    r_set: &[Matrix3<f64>; NUM_CONFIGURATIONS],
    x_sets: &'a [Vec<Vector3<f64>>; NUM_CONFIGURATIONS],
) -> CorrectPoseWithEstimation<'a> {
    // compute the score for the set of points belonging to each configuration,
    // and then return the group with the most points in front of the camera views.
    let mut max_satisfied = (0, 0);
    for i in 0..NUM_CONFIGURATIONS {
        let c = c_set[i];
        let r = r_set[i];
        let x_set = &x_sets[i];

        let score = x_set.iter().fold(0, |score, x| {
            let x_prime_z = r.row(2) * (x - c);
            if x.z > 0.0 && x_prime_z[0] > 0.0 {
                score + 1
            } else {
                score
            }
        });

        if score > max_satisfied.1 {
            max_satisfied = (i, score);
        }
    }

    (
        c_set[max_satisfied.0],
        r_set[max_satisfied.0],
        &x_sets[max_satisfied.0],
    )
}
