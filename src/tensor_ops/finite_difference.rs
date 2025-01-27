use crate::ndarray_ext::NdArray;
use crate::op;
#[cfg(all(feature = "blas", feature = "intel-mkl"))]
use crate::same_type;

#[cfg(all(feature = "blas", feature = "intel-mkl"))]
use crate::tensor_ops::blas_ffi::*;
use crate::tensor_ops::*;
use crate::Float;

use ndarray;
use statrs;

use ndarray::prelude::*;
use ndarray_linalg::Solve;

pub struct FiniteDifference {
    pub order: usize,
    pub accuracy: usize,
    pub step: f64
}

impl<F: Float> op::Op<F> for FiniteDifference {
    fn compute(&self, ctx: &mut op::ComputeContext<F>) -> Result<(), op::OpError> {
        let coeffs =
            finite_difference_coeffs(self.order, self.accuracy).mapv(|x| F::from(x).unwrap());
        if coeffs.len() != ctx.num_inputs() {
            return Err(op::OpError::IncompatibleShape(
                "The length of the coefficients does not match the number of stencil points!"
                    .into(),
            ));
        }

        let denom = F::from(self.step.powi(self.order as i32)).unwrap();

        let mut output = ctx.input(1).mapv(|s| s * coeffs[0]);
        for i in 1..ctx.num_inputs() {
            output
                .iter_mut()
                .zip(ctx.input(i).mapv(|s| s * coeffs[i - 1]).iter())
                .for_each(|(out, &diff)| {
                    *out = *out + diff;
                })
        }

        ctx.append_output(output.mapv(|out| out / denom));
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<F>) {
        panic!("Cannot differentiate the finite difference approximation.");
    }
}

fn central_coeffs(order: usize, accuracy: usize) -> NdArray<f64> {
    let arr = match order {
        1 => match accuracy {
            2 => array![-1. / 2., 0., 1. / 2.],
            4 => array![1. / 12., -2. / 3., 0., 2. / 3., -1. / 12.,],
            6 => array![
                -1. / 60.,
                3. / 20.,
                -3. / 4.,
                0.,
                3. / 4.,
                -3. / 20.,
                1. / 60.,
            ],
            8 => array![
                1. / 280.,
                -4. / 105.,
                1. / 5.,
                -4. / 5.,
                0.,
                4. / 5.,
                -1. / 5.,
                4. / 105.,
                -1. / 280.,
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        2 => match accuracy {
            2 => array![1., -2., 1.],
            4 => array![-1. / 12., 4. / 3., -5. / 2., 4. / 3., -1. / 12.,],
            6 => array![
                1. / 90.,
                -3. / 20.,
                3. / 2.,
                -49. / 18.,
                3. / 2.,
                -3. / 20.,
                1. / 90.,
            ],
            8 => array![
                -1. / 560.,
                8. / 315.,
                -1. / 5.,
                8. / 5.,
                -205. / 72.,
                8. / 5.,
                -1. / 5.,
                8. / 315.,
                -1. / 560.,
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        3 => match accuracy {
            2 => array![-1. / 2., 1., 0., -1., 1. / 2.],
            4 => array![1. / 8., -1., 13. / 8., 0., -13. / 8., 1., -1. / 8.,],
            6 => array![
                -7. / 240.,
                3. / 10.,
                -169. / 120.,
                61. / 30.,
                0.,
                -61. / 30.,
                169. / 60.,
                -2. / 5.,
                7. / 240.,
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        4 => match accuracy {
            2 => array![1., -4., 6., -4., 1.],
            4 => array![-1. / 6., 2., -13. / 2., 28. / 3., -13. / 2., 2., -1. / 6.,],
            6 => array![
                7. / 240.,
                -2. / 5.,
                169. / 60.,
                -122. / 15.,
                91. / 8.,
                -122. / 15.,
                169. / 60.,
                -2. / 5.,
                7. / 240.,
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        5 => match accuracy {
            2 => array![-1. / 2., 2., -5. / 2., 0., 5. / 2., -2., 1. / 2.,],
            4 => array![
                1. / 6.,
                -3. / 2.,
                13. / 3.,
                -29. / 6.,
                0.,
                29. / 6.,
                -13. / 3.,
                3. / 2.,
                -1. / 6.,
            ],
            6 => array![
                -13. / 288.,
                19. / 36.,
                -87. / 32.,
                13. / 2.,
                -323. / 48.,
                0.,
                323. / 48.,
                -13. / 2.,
                -13. / 2.,
                87. / 32.,
                -19. / 36.,
                13. / 288.
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        6 => match accuracy {
            2 => array![1., -6., 15., -20., 15., -6., 1.],
            4 => array![-1. / 4., 3., -13., 29., -75. / 2., 29., -13., 3., -1. / 4.,],
            6 => array![
                13. / 240.,
                -19. / 24.,
                87. / 16.,
                -39. / 2.,
                323. / 8.,
                -1023. / 20.,
                323. / 8.,
                -39. / 2.,
                87. / 16.,
                -19. / 24.,
                13. / 240.
            ],
            _ => panic!(
                "Invalid accuracy {} for derivative of order {}",
                accuracy, order
            ),
        },
        _ => panic!("Invalid derivative order of {} > 6", order),
    };
    arr.into_dyn()
}

fn finite_difference_coeffs(order: usize, accuracy: usize) -> NdArray<f64> {
    if (1 <= order && order <= 2 && accuracy <= 8 && accuracy % 2 == 0)
        || (3 <= order && order <= 6 && accuracy <= 6 && accuracy % 2 == 0)
    {
        return central_coeffs(order, accuracy);
    }
    let p = (2 * ((order + 1) / 2) - 2 + accuracy) / 2;
    let pf = p as f64;
    let mut a = ndarray::Array2::<f64>::zeros(ndarray::Ix2(2 * p + 1, 2 * p + 1));
    let mut b = ndarray::Array1::<f64>::zeros(ndarray::Ix1(2 * p + 1));
    for row in 0..2 * p + 1 {
        for col in 0..2 * p + 1 {
            let v = if col <= p {
                (col as f64) - pf
            } else {
                2. * pf - (col as f64)
            };
            a[[row, col]] = v.powf(row as f64);
        }
    }
    b[order] = statrs::function::factorial::factorial(order as u64);
    a.solve_into(b).unwrap().into_dyn()
}
