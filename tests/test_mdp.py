import torch

from supermimic_hoi.mdp import contact_score, force_scores


def test_force_scores_have_a_threshold_and_saturate():
    force = torch.tensor([0.0, 1.0, 2.5, 5.0, 40.0])
    assert torch.allclose(force_scores(force, 1.0, 5.0), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_contact_score_uses_intended_fingers_and_the_worse_required_hand():
    scores = torch.tensor([[1.0, 1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    none = torch.zeros(1, 10)
    left_two = torch.tensor([[1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    both = torch.tensor([[1.0, 1, 0, 0, 0, 1, 1, 1, 1, 1]])
    assert float(contact_score(scores, none)) == 0.0  # nothing intended, nothing rewarded
    assert float(contact_score(scores, left_two)) == 1.0  # unintended fingers do not dilute the score
    assert float(contact_score(scores, both)) == 0.5  # a perfect left hand cannot hide the right one
    assert float(contact_score(torch.zeros(1, 10), both)) == 0.0
