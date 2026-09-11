"""Explicit, auditable standard-PPO exploration reset for reviewed reward changes."""
import math
import torch

HEADING_OBSERVATION_INDEX = 58


def validate_noise_reset(mode, reward_revision, checkpoint, value):
    if value is None:
        return
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Action std must be finite and strictly positive")
    if mode != "train" or not reward_revision or checkpoint is None:
        raise ValueError("--reset_action_std requires train, --reward_revision, and a checkpoint")


@torch.no_grad()
def reset_action_std(policy, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Action std must be finite and strictly positive")
    representation = policy.noise_std_type
    if representation not in ("scalar", "log"):
        raise ValueError(f"Unsupported standard PPO noise representation: {representation}")
    parameter = policy.std if representation == "scalar" else policy.log_std
    previous = parameter.detach().clone()
    before = {name: p.detach().clone() for name, p in policy.named_parameters()
              if name not in ("std", "log_std")}
    parameter.fill_(value if representation == "scalar" else math.log(value))
    unchanged = all(torch.equal(before[name], p) for name, p in policy.named_parameters()
                    if name in before)
    if not unchanged:
        raise RuntimeError("Exploration reset changed non-noise policy parameters")
    old_std = previous if representation == "scalar" else previous.exp()
    new_std = parameter if representation == "scalar" else parameter.exp()
    return {"representation": representation, "previous_std": old_std.cpu().tolist(),
            "requested_std": value, "applied_std": new_std.detach().cpu().tolist(),
            "non_noise_parameters_unchanged": unchanged}


@torch.no_grad()
def migrate_heading_observation(policy, source_state_dict, input_index=HEADING_OBSERVATION_INDEX):
    """Insert a zero heading column while exactly preserving old parameters."""
    target = policy.state_dict()
    if set(source_state_dict) != set(target):
        missing = sorted(set(target) - set(source_state_dict))
        extra = sorted(set(source_state_dict) - set(target))
        raise ValueError(f"Checkpoint parameter mismatch; missing={missing}, extra={extra}")
    input_keys = ("actor.0.weight", "critic.0.weight")
    migrated = {}
    for key, destination in target.items():
        source = source_state_dict[key]
        if key in input_keys:
            if (source.ndim != 2 or destination.ndim != 2
                    or destination.shape[0] != source.shape[0]
                    or destination.shape[1] != source.shape[1] + 1
                    or not 0 <= input_index <= source.shape[1]):
                raise ValueError(f"Cannot insert heading observation into {key}: "
                                 f"source={tuple(source.shape)}, target={tuple(destination.shape)}")
            expanded = source.new_zeros(destination.shape)
            expanded[:, :input_index] = source[:, :input_index]
            expanded[:, input_index + 1:] = source[:, input_index:]
            migrated[key] = expanded
        else:
            if source.shape != destination.shape:
                raise ValueError(f"Unexpected checkpoint shape for {key}: "
                                 f"source={tuple(source.shape)}, target={tuple(destination.shape)}")
            migrated[key] = source
    result = policy.load_state_dict(migrated, strict=True)
    if isinstance(result, bool):
        if not result:
            raise RuntimeError("Heading migration policy loader did not resume training")
    elif result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Heading migration did not load strictly: {result}")
    loaded = policy.state_dict()
    inserted_zero = all(torch.count_nonzero(loaded[key][:, input_index]).item() == 0
                        for key in input_keys)
    preserved = all(
        torch.equal(loaded[key][:, :input_index].cpu(), source_state_dict[key][:, :input_index].cpu())
        and torch.equal(loaded[key][:, input_index + 1:].cpu(), source_state_dict[key][:, input_index:].cpu())
        for key in input_keys)
    non_input_unchanged = all(torch.equal(loaded[key].cpu(), source_state_dict[key].cpu())
                              for key in target if key not in input_keys)
    if not (inserted_zero and preserved and non_input_unchanged):
        raise RuntimeError("Heading migration did not preserve the source controller")
    return {"feature": "inertial_forward_heading_error_rad", "input_index": input_index,
            "old_input_dim": int(source_state_dict[input_keys[0]].shape[1]),
            "new_input_dim": int(target[input_keys[0]].shape[1]),
            "actor_and_critic_inserted_columns_zero": inserted_zero,
            "old_input_columns_preserved": preserved,
            "non_input_parameters_unchanged": non_input_unchanged}
