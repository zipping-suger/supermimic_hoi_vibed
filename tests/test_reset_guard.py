import pytest
import torch

from supermimic_hoi.reset_guard import nearest_admissible_frame


def test_frames_map_to_the_nearest_admissible_frame_of_their_clip():
    admissible = torch.tensor([1, 0, 0, 0, 1, 1,   0, 1, 0, 1], dtype=torch.bool)
    mapping = nearest_admissible_frame(admissible, torch.tensor([0, 6]), torch.tensor([5, 9]))
    # Clip 0: frame 5 is the last frame and never a reset state. Ties go to the earlier frame.
    assert mapping[:6].tolist() == [0, 0, 0, 4, 4, 4]
    # Clip 1 never borrows a frame from clip 0, and its admissible last frame is excluded too.
    assert mapping[6:].tolist() == [7, 7, 7, 7]


def test_a_clip_without_admissible_frames_is_an_error():
    with pytest.raises(ValueError):
        nearest_admissible_frame(torch.tensor([False, False, True]), torch.tensor([0]), torch.tensor([2]))


def test_overlap_is_measured_per_kind_of_collision_pair(tiny_config):
    from supermimic_hoi.env import make_task
    from supermimic_hoi.reset_guard import OVERLAP_KINDS

    env = make_task(tiny_config).env
    m, depths, summary = env.motion, env.reset_depths, env.reset_summary
    assert depths.shape == (m.num_frames, len(OVERLAP_KINDS)) and (depths >= 0).all()
    # Weave's approach prefix keeps the hands away from the object; the grasp frames overlap it.
    hand_object = depths[:, OVERLAP_KINDS.index("hand_object")]
    assert hand_object[m.clip_start].max() == 0 and hand_object.max() > 0.005
    limit = torch.tensor([tiny_config.reset_overlap_tolerance[kind] for kind in OVERLAP_KINDS])
    admissible = (depths <= limit).all(-1)
    # Every start is admissible, and admissible frames other than a clip's last one are kept as they are.
    kept = admissible.clone()
    kept[m.clip_end] = False
    assert admissible[env.reset_map].all() and (env.reset_map[kept] == kept.nonzero().flatten()).all()
    assert 0 < summary["admitted_contact_frames"] <= summary["contact_frames"]
