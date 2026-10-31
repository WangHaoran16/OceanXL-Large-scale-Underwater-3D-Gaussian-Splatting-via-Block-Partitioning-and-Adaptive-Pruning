"""Underwater-compensated Densification (UCD), paper Sec. 3.2, Eq. 2.

Gaussians are grouped by camera-space depth z into near (z < z_near), mid and far (z >= z_far),
with z_near = z_max / 3 and z_far = 2 z_max / 3 (z_max: deepest visible Gaussian in the view).
Their densification gradients are weighted by 1, f_mid, f_far. Densification is not
differentiable, so (f_mid, f_far) are learned with gradient-free SPSA:

  * each densification interval runs with one perturbed setting theta +/- c_k * delta
    (delta in {-1, +1}^2); the two intervals of a pair use opposite signs;
  * the objective is the mean training L1 over the interval that follows a densification;
  * after a pair: g = (L+ - L-) / (2 c_k) * delta / max(L+, L-)  (relative: the loss scale
    shrinks during training), theta <- theta - a_k * g;
  * theta is projected to f_min <= f_mid <= f_far <= f_max (paper: f_far >= f_mid).

The +/- order alternates between pairs and delta is random, so the downward drift of the
training loss inside a pair cancels in expectation. A pair that contains an opacity reset is
discarded, since the reset makes the loss jump.
"""
import random


class UCDSPSA:
    def __init__(self, f_init=1.0, f_min=1.0, f_max=4.0, a=2.0, c=0.25, alpha=0.602, gamma=0.101,
                 seed=0, log_path=None):
        self.theta = [f_init, f_init]
        self.f_min, self.f_max = f_min, f_max
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma
        self.rng = random.Random(seed)
        self.k = 0  # completed SPSA updates
        self.log_path = log_path
        self._new_pair()

    # ---- SPSA bookkeeping ------------------------------------------------------------------
    def _project(self, t):
        t = [min(max(v, self.f_min), self.f_max) for v in t]
        if t[1] < t[0]:  # enforce f_far >= f_mid
            m = 0.5 * (t[0] + t[1])
            t = [m, m]
        return t

    def _new_pair(self):
        self.delta = [self.rng.choice((-1.0, 1.0)) for _ in range(2)]
        self.ck = self.c / (self.k + 1) ** self.gamma
        self.signs = (1.0, -1.0) if self.k % 2 == 0 else (-1.0, 1.0)
        self.phase = 0  # index into self.signs
        self.pair_losses = {}
        self._reset_interval()

    def _reset_interval(self):
        self.loss_sum = 0.0
        self.loss_n = 0
        self.dirty = False

    def current(self):
        """(f_mid, f_far) to use in the running interval."""
        s = self.signs[self.phase]
        t = self._project([th + s * self.ck * d for th, d in zip(self.theta, self.delta)])
        return t[0], t[1]

    def record(self, l1):
        self.loss_sum += float(l1)
        self.loss_n += 1

    def mark_dirty(self):
        """The running interval contains an opacity reset: its loss is not comparable."""
        self.dirty = True

    def end_interval(self, iteration):
        """Call right after each densification step."""
        if self.dirty or self.loss_n == 0:
            self._new_pair()  # throw away the whole pair
            return
        self.pair_losses[self.signs[self.phase]] = self.loss_sum / self.loss_n
        if self.phase == 0:
            self.phase = 1
            self._reset_interval()
            return
        lp, lm = self.pair_losses[1.0], self.pair_losses[-1.0]
        ak = self.a / (self.k + 1) ** self.alpha
        scale = (lp - lm) / (2 * self.ck) / max(lp, lm, 1e-12)
        self.theta = self._project([th - ak * scale * d for th, d in zip(self.theta, self.delta)])
        self.k += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write("{} f_mid={:.4f} f_far={:.4f} L+={:.6f} L-={:.6f}\n".format(
                    iteration, self.theta[0], self.theta[1], lp, lm))
        self._new_pair()

    def state(self):
        return self.theta[0], self.theta[1]

    # ---- per-Gaussian weights ---------------------------------------------------------------
    @staticmethod
    def depth_weights(xyz, viewpoint_cam, visible, f_mid, f_far):
        """w_i of Eq. 2 for every Gaussian in this view, shape (N, 1)."""
        import torch
        W = viewpoint_cam.world_view_transform  # 3DGS convention: p_view = [p, 1] @ W
        z = xyz.detach() @ W[:3, 2] + W[3, 2]
        w = torch.ones_like(z)
        if visible.any():
            z_max = z[visible].max()
            w = torch.where(z >= z_max / 3, torch.full_like(z, f_mid), w)
            w = torch.where(z >= 2 * z_max / 3, torch.full_like(z, f_far), w)
        return w.unsqueeze(1)


def spsa_sanity_check(steps=4000):
    """Toy objective L = (f_mid - 2)^2 + (f_far - 3)^2 + 1 (+ noise): theta should reach (2, 3)."""
    s = UCDSPSA(f_init=1.0, a=1.0, c=0.2)
    noise = random.Random(1)
    for it in range(steps):
        f_mid, f_far = s.current()
        s.record((f_mid - 2) ** 2 + (f_far - 3) ** 2 + 1 + 0.01 * noise.gauss(0, 1))
        s.end_interval(it)
    return s.state()


if __name__ == "__main__":
    print("SPSA toy optimum (2, 3) ->", tuple(round(v, 3) for v in spsa_sanity_check()))
    print("project([3, 2]) ->", UCDSPSA()._project([3.0, 2.0]), "(f_far >= f_mid)")
