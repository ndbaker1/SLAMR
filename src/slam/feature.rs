use nalgebra::Vector2;

use crate::algorithms::brief::BinaryDescriptor;

/// Feature object which holds a coordinate/pixel on a page
/// and tries to handle a generic descriptor
#[derive(Clone)]
pub struct Feature<Descriptor> {
    pub keypoint: Vector2<u32>,
    pub descriptor: Descriptor,
}

impl<const N: usize> Default for Feature<BinaryDescriptor<N>> {
    fn default() -> Self {
        Self {
            keypoint: Vector2::default(),
            descriptor: [0; N],
        }
    }
}
