import itertools

import pytest
import torch

from sapo_core.distributions import FeasibleBernoulli, NonemptyBernoulli
from sapo_core.policy import Actor, Observation


def example(edges=None, capacity=None, residency=None):
    return Observation(
        state=torch.tensor([0.2, -0.4, 0.1]),
        node_features=torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5]]),
        device_features=torch.tensor([[1., 0.], [0., 1.]]),
        head_id=torch.tensor([0, 1, -1]),
        edges=torch.tensor([[0, 2], [1, 2]] if edges is None else edges, dtype=torch.long).reshape(-1, 2),
        residency=torch.tensor([[1, 1], [1, 1], [1, 1]] if residency is None else residency, dtype=torch.bool),
        memory_cost=torch.tensor([1., 1., 1.]),
        capacity=torch.tensor([2., 2.] if capacity is None else capacity),
    )


def test_nonempty_gate_normalization_and_extreme_logits():
    for logits in (torch.tensor([-1., 0.4, 2.], dtype=torch.float64),
                   torch.tensor([-1000., -1001., -1002.], dtype=torch.float64)):
        dist = NonemptyBernoulli(logits)
        gates = [torch.tensor(g, dtype=torch.bool) for g in itertools.product((False, True), repeat=3) if any(g)]
        assert torch.stack([dist.log_prob(g).exp() for g in gates]).sum().item() == pytest.approx(1, abs=1e-6)
        for _ in range(30):
            assert bool(dist.sample().any())
        assert torch.isfinite(dist.entropy())


def test_gate_entropy_and_gradient_match_enumeration():
    logits = torch.tensor([-1.2, 0.3, 1.1], dtype=torch.float64, requires_grad=True)
    dist = NonemptyBernoulli(logits)
    gates = [torch.tensor(g, dtype=torch.bool) for g in itertools.product((False, True), repeat=3) if any(g)]
    log_prob = torch.stack([dist.log_prob(g) for g in gates])
    probability = log_prob.exp()
    enumerated = -(probability * log_prob).sum()
    analytic = dist.entropy()
    assert torch.allclose(analytic, enumerated, atol=1e-12)
    exact_grad = torch.autograd.grad(enumerated, logits, retain_graph=True)[0]
    analytic_grad = torch.autograd.grad(analytic, logits)[0]
    assert torch.allclose(analytic_grad, exact_grad, atol=1e-12)


def test_sample_and_teacher_forcing_match_and_gradients():
    torch.manual_seed(9)
    actor = Actor(3, 2, 2, 2, 16)
    obs = example()
    for _ in range(20):
        action = actor.sample_action(obs)
        replay = actor.evaluate_actions(obs, action.gate, action.mapping)
        assert torch.allclose(action.log_prob, replay, atol=1e-6)
        assert int((action.mapping != -1).sum()) == int((obs.head_id.eq(-1) | action.gate[obs.head_id.clamp(min=0)]).sum())
    (-action.log_prob).backward()
    assert actor.gate_head.weight.grad is not None
    assert torch.isfinite(actor.gate_head.weight.grad).all()
    assert actor.node_encoder.weight.grad is not None
    assert torch.isfinite(actor.node_encoder.weight.grad).all()


def test_full_joint_probability_normalizes():
    actor = Actor(3, 2, 2, 2, 12)
    obs = example(capacity=[1., 2.], residency=[[1, 1], [1, 0], [0, 1]])
    total = 0.
    valid_actions = 0
    for gate_bits in ((True, False), (False, True), (True, True)):
        gate = torch.tensor(gate_bits)
        for assignment in itertools.product((-1, 0, 1), repeat=3):
            mapping = torch.tensor(assignment)
            try:
                total += actor.evaluate_actions(obs, gate, mapping).exp().item()
                valid_actions += 1
            except ValueError:
                pass
    assert valid_actions > 3
    assert total == pytest.approx(1., abs=1e-6)


def test_restricted_gate_support_normalizes_joint_and_has_gradients():
    torch.manual_seed(7)
    actor = Actor(3, 2, 2, 2, 12)
    with torch.no_grad():
        actor.gate_head.weight.zero_()
        actor.gate_head.bias.copy_(torch.tensor([0.7, -0.2]))
    # The always-active node consumes one of two one-unit slots. Either
    # singleton head fits, but both heads together do not.
    obs = example(capacity=[1., 1.])
    mass_by_gate = {}
    all_scores = []
    for gate_bits in ((True, False), (False, True), (True, True)):
        gate = torch.tensor(gate_bits)
        mass = 0.
        for assignment in itertools.product((-1, 0, 1), repeat=3):
            try:
                score = actor.evaluate_actions(obs, gate, torch.tensor(assignment))
            except ValueError:
                continue
            mass += score.exp().item()
            all_scores.append(score)
        mass_by_gate[gate_bits] = mass
    expected_gate_mass = torch.softmax(torch.tensor([0.7, -0.2]), 0)
    assert mass_by_gate[(True, False)] == pytest.approx(expected_gate_mass[0].item(), abs=1e-6)
    assert mass_by_gate[(False, True)] == pytest.approx(expected_gate_mass[1].item(), abs=1e-6)
    assert mass_by_gate[(True, True)] == 0.
    assert sum(mass_by_gate.values()) == pytest.approx(1., abs=1e-6)
    actor.zero_grad()
    (-all_scores[0]).backward()
    assert torch.isfinite(actor.gate_head.bias.grad).all()
    assert actor.gate_head.bias.grad.abs().sum() > 0
    for _ in range(30):
        action = actor.sample_action(obs)
        assert int(action.gate.sum()) == 1
        assert torch.allclose(action.log_prob, actor.evaluate_actions(obs, action.gate, action.mapping))


def test_restricted_distribution_probabilities_and_entropy():
    logits = torch.tensor([1.2, -0.4], dtype=torch.float64, requires_grad=True)
    support = torch.tensor([[True, False], [False, True]])
    dist = FeasibleBernoulli(logits, support)
    probability = torch.stack([dist.log_prob(g).exp() for g in support])
    assert torch.allclose(probability, torch.softmax(logits, 0), atol=1e-12)
    entropy = -(probability * probability.log()).sum()
    assert torch.allclose(dist.entropy(), entropy, atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(entropy, logits)[0]).all()
    with pytest.raises(ValueError, match="duplicate gates"):
        FeasibleBernoulli(logits.detach(), torch.tensor([[True, False], [True, False]]))


def test_deterministic_action_is_repeatable_and_replayable():
    actor = Actor(3, 2, 2, 2, 12)
    for obs in (example(), example(capacity=[1., 1.])):
        actions = [actor.sample_action(obs, deterministic=True) for _ in range(3)]
        assert all(torch.equal(actions[0].gate, action.gate) for action in actions[1:])
        assert all(torch.equal(actions[0].mapping, action.mapping) for action in actions[1:])
        assert torch.allclose(actions[0].log_prob,
                              actor.evaluate_actions(obs, actions[0].gate, actions[0].mapping))


def test_full_graph_feasible_keeps_original_nonempty_bernoulli_gate():
    actor = Actor(3, 2, 2, 2, 12)
    obs = example()
    logits = actor.gate_head(actor.state_encoder(obs.state))
    original_gate = NonemptyBernoulli(logits)
    for gate_bits in ((True, False), (False, True), (True, True)):
        gate = torch.tensor(gate_bits)
        mapping_mass = 0.
        for assignment in itertools.product((-1, 0, 1), repeat=3):
            try:
                mapping_mass += actor.evaluate_actions(obs, gate, torch.tensor(assignment)).exp().item()
            except ValueError:
                pass
        assert mapping_mass == pytest.approx(original_gate.log_prob(gate).exp().item(), abs=1e-6)


def test_fresh_sample_multinode_gradient_has_no_inplace_error():
    actor = Actor(3, 2, 2, 2, 12)
    obs = example()
    # Use a fresh sampled action in each backward pass, with every operator
    # active so the decoder has several autoregressive prefixes.
    with torch.no_grad():
        actor.gate_head.weight.zero_()
        actor.gate_head.bias.fill_(10.)
    for _ in range(5):
        actor.zero_grad()
        action = actor.sample_action(obs)
        (-action.log_prob).backward()
        assert torch.isfinite(actor.decoder.weight_ih.grad).all()
        assert torch.isfinite(actor.gat.query.weight.grad).all()


def test_lookahead_masks_dead_end_and_residency():
    torch.manual_seed(3)
    actor = Actor(3, 2, 2, 2, 12)
    obs = example(capacity=[1., 2.], residency=[[1, 1], [1, 0], [0, 1]])
    gate = torch.tensor([True, True])
    # Node 0 must go to device 1 so node 1 can use the sole device-0 slot.
    feasible = torch.tensor([1, 0, 1])
    score = actor.evaluate_actions(obs, gate, feasible)
    assert torch.isfinite(score)
    with pytest.raises(ValueError, match="infeasible"):
        actor.evaluate_actions(obs, gate, torch.tensor([0, 0, 1]))
    for _ in range(30):
        action = actor.sample_action(obs)
        if bool(action.gate[0] and action.gate[1]):
            assert action.mapping.tolist() == [1, 0, 1]


def test_invalid_actions_and_impossible_scene():
    actor = Actor(3, 2, 2, 2, 12)
    obs = example()
    with pytest.raises(ValueError, match="all-zero"):
        actor.evaluate_actions(obs, torch.tensor([False, False]), torch.tensor([-1, -1, 0]))
    with pytest.raises(ValueError, match="inactive"):
        actor.evaluate_actions(obs, torch.tensor([True, False]), torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="active"):
        actor.evaluate_actions(obs, torch.tensor([True, False]), torch.tensor([-1, -1, 1]))
    impossible = example(capacity=[0., 0.])
    with pytest.raises(ValueError, match="no nonempty gate"):
        actor.sample_action(impossible)
    cyclic = example(edges=[[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="DAG"):
        actor.sample_action(cyclic)
    duplicate_edge = example(edges=[[0, 2], [0, 2]])
    with pytest.raises(ValueError, match="duplicate DAG edges"):
        actor.sample_action(duplicate_edge)


def test_input_contract_rejects_missing_heads_empty_graph_and_memory_fractions():
    from dataclasses import replace

    actor = Actor(3, 2, 2, 2, 12)
    obs = example()
    with pytest.raises(ValueError, match="at least one node"):
        actor.sample_action(replace(obs, node_features=torch.empty(0, 2)))
    with pytest.raises(ValueError, match="2-D"):
        actor.sample_action(replace(obs, node_features=torch.empty(3)))
    with pytest.raises(ValueError, match="has no operator node"):
        actor.sample_action(replace(obs, head_id=torch.tensor([0, 0, -1])))
    with pytest.raises(ValueError, match="integer memory units"):
        actor.sample_action(replace(obs, memory_cost=torch.tensor([0.5, 1., 1.])))
    with pytest.raises(ValueError, match="floating-point dtype"):
        actor.sample_action(replace(obs, state=obs.state.long()))
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="same device"):
            actor.sample_action(replace(obs, state=obs.state.cuda()))


def test_gat_receives_dependency_information():
    torch.manual_seed(4)
    actor = Actor(3, 2, 2, 2, 12)
    features = actor.node_encoder(example().node_features)
    without = actor.gat(features, [0, 1, 2], [[], [], []])[2]
    with_edges = actor.gat(features, [0, 1, 2], [[], [], [0, 1]])[2]
    assert not torch.allclose(without, with_edges)
