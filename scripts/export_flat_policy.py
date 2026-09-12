"""Export a saved 68D flat-ground actor on CPU, with ONNX Runtime parity checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from rsl_rl.modules import ActorCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    config = yaml.safe_load((checkpoint.parent / 'agent.yaml').read_text())
    provenance = json.loads((checkpoint.parent / 'provenance.json').read_text())
    policy_config = dict(config['policy'])
    assert policy_config.pop('class_name') == 'ActorCritic'
    assert not policy_config.get('actor_obs_normalization', False)
    assert config['obs_groups']['policy'] == ['policy']
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy = ActorCritic({'policy': torch.zeros(1, 68)}, config['obs_groups'], 12,
                         **policy_config).eval()
    policy.load_state_dict(saved['model_state_dict'], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / 'policy.onnx'
    torch.onnx.export(policy.actor, torch.zeros(1, 68), str(output),
                      input_names=['obs'], output_names=['actions'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(output)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(2)
    max_error = 0.
    for scale in (0., .1, 1., 5.):
        for _ in range(32):
            obs = (rng.standard_normal((1, 68)) * scale).astype(np.float32)
            with torch.no_grad():
                reference = policy.act_inference({'policy': torch.from_numpy(obs)}).numpy()
            actual = session.run(['actions'], {'obs': obs})[0]
            np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - reference))))
    metadata = {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'checkpoint_iteration': saved['iter'], 'seed': provenance['seed'],
        'fresh_training': provenance.get('fresh_training'),
        'training_kind': provenance.get('training_kind', 'fresh'),
        'parent_checkpoint_sha256': provenance.get('parent_checkpoint_sha256'),
        'deployment_domain_randomization': provenance.get('deployment_domain_randomization', False),
        'task_sha256': provenance['task_sha256'],
        'input': {'name': 'obs', 'shape': [1, 68], 'dtype': 'float32'},
        'output': {'name': 'actions', 'shape': [1, 12], 'dtype': 'float32'},
        'policy_rate_hz': 50, 'actor_observation_normalization': False,
        'action_mapping': 'q_target = default_joint_pos + 0.25 * clip(actions, -5, 5)',
        'validation': {'inputs': 128, 'provider': 'CPUExecutionProvider',
                       'max_absolute_error': max_error},
        'note': 'Integration artifact; this export does not establish hardware readiness.'}
    (args.output / 'export_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'onnx': str(output.resolve()), 'max_absolute_error': max_error}, indent=2))


if __name__ == '__main__':
    main()
