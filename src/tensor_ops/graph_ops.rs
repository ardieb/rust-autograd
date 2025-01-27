


use crate::op;


use crate::Float;

pub(crate) struct ControlDependency;

impl<F: Float> op::Op<F> for ControlDependency {
    // Reuse the 1st input
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0);
        ctx.append_output_view(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        for _ in 0..ctx.num_inputs() {
            ctx.append_input_grad(None);
        }
    }
}
