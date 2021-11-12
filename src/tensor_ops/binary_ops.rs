use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::Float;
use crate::Graph;
use ndarray;
use std::mem;

pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;
pub struct MaybeReduce;
pub struct MaybeBroadcast;

#[cfg(feature = "mkl")]
macro_rules! bin_op_same_shape {
    ($vms_op:ident, $vmd_op:ident, $std_op:tt, $a:expr, $b:expr) => {
        unsafe {
            if same_type::<T, f32>() {
                let mut y = Vec::with_capacity($a.len());
                $vms_op($a.len() as MklInt, $a.as_ptr() as *const f32, $b.as_ptr() as *const f32, y.as_mut_ptr() as *mut f32);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else if same_type::<T, f64>() {
                let mut y = Vec::with_capacity($a.len());
                $vmd_op($a.len() as MklInt, $a.as_ptr() as *const f64, $b.as_ptr() as *const f64, y.as_mut_ptr() as *mut f64);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else {
                $a $std_op $b
            }
        }
    };
}

impl<T: Float> op::Op<T> for MaybeReduce {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let input = ctx.input(0);
        let target_shape__ = crate::ndarray_ext::as_shape(&ctx.input(1));
        let target_shape = target_shape__.as_slice();
        let input_shape = input.shape();

        if target_shape == input_shape {
            // The case where forward path didn't cause broadcast.
            ctx.append_output_view(input.clone());
        } else {
            // Broadcast occurred. We need reduction of `input`.
            // First, handle the case where `input` is scalar.
            let target_shape_is_scalar = crate::ndarray_ext::is_scalar_shape(target_shape);
            let target_shape_ = if target_shape_is_scalar {
                vec![1; input_shape.len()]
            } else {
                target_shape.to_vec()
            };

            // Reduce each dim as necessary
            let mut folded: Option<NdArray<T>> = None;
            for (i, (x_axis, gy_axis)) in target_shape_.iter().zip(input_shape).enumerate() {
                if x_axis < gy_axis {
                    if *x_axis == 1 {
                        // `fold_axis` squashes the axis automatically.
                        let axis = ndarray::Axis(if target_shape_is_scalar { 0 } else { i });
                        let ret = match folded {
                            Some(ref a) => a.fold_axis(axis.clone(), T::zero(), |&a, &b| a + b),
                            None => input.fold_axis(axis.clone(), T::zero(), |&a, &b| a + b),
                        };
                        if target_shape_is_scalar {
                            mem::swap(&mut folded, &mut Some(ret));
                        } else {
                            // Expands squashed axis.
                            mem::swap(
                                &mut folded,
                                &mut Some(crate::ndarray_ext::expand_dims(ret, i)),
                            );
                        }
                    } else {
                        return Err(op::OpError::IncompatibleShape(
                            "Incorrect gradient shape".to_string(),
                        ));
                    }
                }
                // case of x_axis < gy_axis: unreachable
                // case of x_axis == gy_axis: nothing to do
            }
            // TODO
            let ret = folded.unwrap();
            debug_assert_eq!(target_shape, ret.shape());
            ctx.append_output(ret);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(&ctx.output_grad(), false)
            .append_input(&shape(ctx.input(0)), false)
            .build(MaybeBroadcast);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

// Do broadcast if necessary.
impl<T: Float> op::Op<T> for MaybeBroadcast {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let target_shape_ = ctx.input(1);
        let target_shape_ = crate::ndarray_ext::as_shape(&target_shape_);
        let target_shape = target_shape_.as_slice();

        let raw_input = ctx.input(0);
        if raw_input.shape() == target_shape {
            ctx.append_output_view(raw_input);
            return Ok(());
        }

        // make broadcast dims if needed
        let input_is_scalar = crate::ndarray_ext::is_scalar_shape(raw_input.shape());
        let input = if input_is_scalar {
            raw_input.into_shape(vec![1; target_shape.len()]).unwrap()
        } else {
            raw_input
        };

        // do broadcast
        if let Some(ret) = input.broadcast(target_shape) {
            ctx.append_output(ret.to_owned());
            Ok(())
        } else {
            Err(op::OpError::IncompatibleShape(
                "PreprocessBinOpGradGrad: Can't broadcast.".to_string(),
            ))
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let gx = maybe_reduce(&shape(ctx.input(0)), &ctx.output_grad(), g);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for AddOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = add_forward(&ctx.input(0), &ctx.input(1));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy0 = maybe_reduce(shape0, &gy, g);
        let gy1 = maybe_reduce(shape1, &gy, g);
        ctx.append_input_grad(Some(gy0));
        ctx.append_input_grad(Some(gy1));
    }
}

impl<T: Float> op::Op<T> for SubOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0: &[usize] = x0.shape();
        let shape1: &[usize] = x1.shape();
        let ret = if shape0.len() == 0 {
            // is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem - a)
        } else if shape0 == shape1 {
            #[cfg(feature = "mkl")]
            {
                use crate::{same_type, tensor_ops::blas_ffi::*};
                bin_op_same_shape!(vsSub, vdSub, -, x0, x1)
            }
            #[cfg(not(feature = "mkl"))]
            {
                x0 - x1
            }
        } else {
            x0 - x1
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy = &ctx.output_grad();
        let gy0 = maybe_reduce(shape0, gy, g);
        let gy1 = maybe_reduce(shape1, gy, g);
        ctx.append_input_grad(Some(gy0));
        ctx.append_input_grad(Some(neg(&gy1)));
    }
}

impl<T: Float> op::Op<T> for MulOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);
        let ret = mul_forward(&a, &b);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let graph = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);

        let shape0 = &shape(x0);
        let shape1 = &shape(x1);

        let gy = ctx.output_grad();

        let gx0 = gy * x1;
        let gx1 = gy * x0;

        let gx0 = maybe_reduce(shape0, &gx0, graph);
        let gx1 = maybe_reduce(shape1, &gx1, graph);

        ctx.append_input_grad(Some(gx0));
        ctx.append_input_grad(Some(gx1));
    }
}

impl<T: Float> op::Op<T> for DivOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0 = x0.shape();
        let shape1 = x1.shape();
        let ret = if shape0 == shape1 {
            #[cfg(feature = "mkl")]
            {
                use crate::{same_type, tensor_ops::blas_ffi::*};
                bin_op_same_shape!(vsDiv, vdDiv, /, x0, x1)
            }
            #[cfg(not(feature = "mkl"))]
            {
                x0 / x1
            }
        } else {
            x0 / x1
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy = ctx.output_grad();

        let gx0 = gy / x1;
        let gx1 = neg(x0) * pow(x1, T::from(-2.).unwrap()) * gy;

        let gx0 = maybe_reduce(shape0, &gx0, g);
        let gx1 = maybe_reduce(shape1, &gx1, g);

        ctx.append_input_grad(Some(gx0));
        ctx.append_input_grad(Some(gx1));
    }
}

fn maybe_reduce<'g, T: Float>(
    target_shape: &Tensor<'g, T>,
    x: &Tensor<'g, T>,
    graph: &'g Graph<T>,
) -> Tensor<'g, T> {
    Tensor::builder(graph)
        .append_input(x, false)
        .append_input(target_shape, false)
        .set_shape(target_shape)
        .build(MaybeReduce)
}

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt, $vms_op:ident, $vmd_op:ident) => {
        fn $forward_name<'v, T: Float>(x0: &NdArrayView<'v, T>, x1: &NdArrayView<'v, T>) -> NdArray<T>
        {
            let shape0: &[usize] = x0.shape();
            let shape1: &[usize] = x1.shape();
            let scalar_shape: &[usize] = &[];
            let scalar_shape1 = &[0];

            let x0_is_scalar = shape0 == scalar_shape || shape0 == scalar_shape1;
            let x1_is_scalar = shape1 == scalar_shape || shape1 == scalar_shape1;

            if x0_is_scalar && !x1_is_scalar {
                let elem = x0[ndarray::IxDyn(&[])];
                x1.map(move |&a| a $bin_op elem)
            } else if x1_is_scalar && !x0_is_scalar {
                let elem = x1[ndarray::IxDyn(&[])];
                x0.map(move |&a| a $bin_op elem )
            } else if !x0_is_scalar && !x1_is_scalar {
                let len0: usize = shape0.iter().product();
                let len1: usize = shape1.iter().product();
                if len0 > len1 {
                    x0 $bin_op x1
                } else {
                    // tensor vs tensor (same shapes)
                    #[cfg(feature = "mkl")]
                    {
                        use crate::{ops::mkl_ffi::*, same_type};
                        bin_op_same_shape!($vms_op, $vmd_op, $bin_op, x0, x1)
                    }
                    #[cfg(not(feature = "mkl"))] {
                        x0 $bin_op x1
                    }
                }
            } else {
                // scalar vs scalar
                x0 $bin_op x1
            }
        }
    };
}

impl_bin_op_forward!(add_forward, +, vsAdd, vdAdd);
impl_bin_op_forward!(mul_forward, *, vsMul, vdMul);
