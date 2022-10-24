use nalgebra::{Matrix3, Matrix3x4, Matrix4, RowVector4, Vector2, Vector3};

pub fn triangulate_linear(
    camera_intrinsics: &Matrix3<f64>,
    c1: &Vector3<f64>,
    r1: &Matrix3<f64>,
    c2: &Vector3<f64>,
    r2: &Matrix3<f64>,
    x1_set: &Vec<Vector2<f64>>,
    x2_set: &Vec<Vector2<f64>>,
) -> Vec<Vector3<f64>> {
    let camera_matrix1 = camera_intrinsics
        * r1
        * Matrix3x4::from_iterator(Matrix3::identity().iter().chain(c1.iter()).copied());

    let camera_matrix2 = camera_intrinsics
        * r2
        * Matrix3x4::from_iterator(Matrix3::identity().iter().chain(c2.iter()).copied());

    // here we are assuming that:
    // x1_set.len() == x2_set.len()
    (0..x1_set.len())
        .map(|i| {
            let (x1, x2) = (x1_set[i], x2_set[i]);

            let matrix_a = Matrix4::from_rows(&[
                RowVector4::new(
                    x1.y * camera_matrix1.m31 - camera_matrix1.m21,
                    x1.y * camera_matrix1.m32 - camera_matrix1.m22,
                    x1.y * camera_matrix1.m33 - camera_matrix1.m23,
                    x1.y * camera_matrix1.m34 - camera_matrix1.m24,
                ),
                RowVector4::new(
                    x1.x * camera_matrix1.m31 - camera_matrix1.m11,
                    x1.x * camera_matrix1.m32 - camera_matrix1.m12,
                    x1.x * camera_matrix1.m33 - camera_matrix1.m13,
                    x1.x * camera_matrix1.m34 - camera_matrix1.m14,
                ),
                RowVector4::new(
                    x2.y * camera_matrix2.m31 - camera_matrix2.m21,
                    x2.y * camera_matrix2.m32 - camera_matrix2.m22,
                    x2.y * camera_matrix2.m33 - camera_matrix2.m23,
                    x2.y * camera_matrix2.m34 - camera_matrix2.m24,
                ),
                RowVector4::new(
                    x2.x * camera_matrix2.m31 - camera_matrix2.m11,
                    x2.x * camera_matrix2.m32 - camera_matrix2.m12,
                    x2.x * camera_matrix2.m33 - camera_matrix2.m13,
                    x2.x * camera_matrix2.m34 - camera_matrix2.m14,
                ),
            ]);

            // V_t obtained from SVD of matrix A
            let matrix_v_t = matrix_a.svd(false, true).v_t.unwrap();

            // convert to homogenous coordinates and back into euclidean
            Vector3::from_iterator(matrix_v_t.row(3).iter().cloned()) / matrix_v_t.row(3)[3]
        })
        .collect()
}

#[allow(unused)] // TODO
pub fn triangulate_nonlinear(
    camera_intrinsics: &Matrix3<f64>,
    c1: &Vector3<f64>,
    r1: &Matrix3<f64>,
    c2: &Vector3<f64>,
    r2: &Matrix3<f64>,
    x1: Vec<(u32, u32)>,
    x2: Vec<(u32, u32)>,
) -> Vec<Vector3<f64>> {
    todo!()
}
