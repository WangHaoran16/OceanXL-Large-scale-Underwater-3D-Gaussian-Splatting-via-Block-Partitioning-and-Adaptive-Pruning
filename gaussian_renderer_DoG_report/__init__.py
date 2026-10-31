"""Evaluation renderer: the same DoG rasterizer as training (dog-gaussian-rasterization, built as
diff_gaussian_rasterization_DoG). Kept as a separate name because training_report and the FPS
script import it; it used to call diff_gaussian_rasterization_DoG_test, which expects three
scaling factors per Gaussian and reads out of bounds on these models.
"""
from gaussian_renderer_DoG import render_DoG


def render_DoG_report(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, separate_sh=True,
                      override_color=None, use_trained_exp=False, render_size=None):
    return render_DoG(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier,
                      separate_sh=separate_sh, override_color=override_color,
                      use_trained_exp=use_trained_exp, render_size=render_size)
