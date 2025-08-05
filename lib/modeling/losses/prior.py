from lib.kits.basic import *

# ============= #
#     Utils     #
# ============= #


def soft_bound_loss(x, low, up):
    '''
    Softly penalize the violation of the lower and upper bounds.
    PROBLEMS: for joints like legs, whose normal pose is near the boundary (standing person tend to have zero rotation but the limitation is zero-bounded, which encourage the leg to bend somehow).

    ### Args:
    - x: torch.tensor
        - shape = (B, Q), where Q is the number of components.
    - low: torch.tensor, lower bound.
        - shape = (Q,)
        - Lower bound.
    - up: (Q,)
        - shape = (Q,)
        - Upper bound.

    ### Returns:
    - loss: torch.tensor
        - shape = (B,)
    '''
    B = len(x)
    loss = torch.exp(low[None] - x).pow(2) + torch.exp(x - up[None]).pow(2)  # (B, Q)
    return loss  # (B,)


def softer_bound_loss(x, low, up):
    '''
    Softly penalize the violation of the lower and upper bounds. This loss won't penalize so hard when the
    value exceed the bound by a small margin (half of up - low), but it's friendly to the case when the common
    case is not centered at the middle of the bound. (E.g., the rotation of knee is more likely to be at zero
    when some one is standing straight, but zero is the lower bound.)

    ### Args:
    - x: torch.tensor, (B, Q), where Q is the number of components.
    - low: torch.tensor, (Q,)
        - Lower bound.
    - up: torch.tensor, (Q,)
        - Upper bound.

    ### Returns:
    - loss: torch.tensor, (B,)
    '''
    B = len(x)
    expand = (up - low) / 2
    loss = torch.exp((low[None] - expand) - x).pow(2) + torch.exp(x - (up[None] + expand)).pow(2)  # (B, Q)
    return loss  # (B,)


def softest_bound_loss(x, low, up):
    '''
    Softly penalize the violation of the lower and upper bounds. This loss won't penalize so hard when the
    value exceed the bound by a small margin (half of up - low), but it's friendly to the case when the common
    case is not centered at the middle of the bound. (E.g., the rotation of knee is more likely to be at zero
    when some one is standing straight, but zero is the lower bound.)

    ### Args:
    - x: torch.tensor, (B, Q), where Q is the number of components.
    - low: torch.tensor, (Q,)
        - Lower bound.
    - up: torch.tensor, (Q,)
        - Upper bound.

    ### Returns:
    - loss: torch.tensor, (B,)
    '''
    B = len(x)
    expand = (up - low) / 2
    lower_loss = torch.exp((low[None] - expand) - x).pow(2) - 1  # (B, Q)
    upper_loss = torch.exp(x - (up[None] + expand)).pow(2) - 1  # (B, Q)
    lower_loss = torch.where(lower_loss < 0, 0, lower_loss)
    upper_loss = torch.where(upper_loss < 0, 0, upper_loss)
    loss = lower_loss + upper_loss
    return loss  # (B,)


# ============= #
#     Loss      #
# ============= #


def compute_poses_angle_prior_loss(poses):
    '''
    Some components have upper and lower bound, use exponential loss to apply soft limitation.

    ### Args
    - poses: torch.tensor, (B, 46)

    ### Returns
    - loss: torch.tensor, (,)
    '''
    from lib.body_models.skel_utils.limits import SKEL_LIM_QIDS, SKEL_LIM_BOUNDS

    device = poses.device
    # loss = softer_bound_loss(
    # loss = softest_bound_loss(
    loss = soft_bound_loss(
            x   = poses[:, SKEL_LIM_QIDS],
            low = SKEL_LIM_BOUNDS[:, 0].to(device),
            up  = SKEL_LIM_BOUNDS[:, 1].to(device),
        ) # (,)

    return loss
