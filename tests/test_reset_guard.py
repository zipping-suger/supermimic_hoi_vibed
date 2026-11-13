import pytest
import torch

from supermimic_hoi.reset_guard import nearest_admissible_frame


def test_frames_map_to_the_nearest_admissible_frame_of_their_clip():
    depths = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.0, 0.0,   0.5, 0.0, 0.5, 0.0])
    mapping = nearest_admissible_frame(depths, torch.tensor([0, 6]), torch.tensor([5, 9]), tolerance=0.1)
    # Clip 0: frame 5 is the last frame and never a reset state. Ties go to the earlier frame.
    assert mapping[:6].tolist() == [0, 0, 0, 4, 4, 4]
    # Clip 1 never borrows a frame from clip 0, and its admissible last frame is excluded too.
    assert mapping[6:].tolist() == [7, 7, 7, 7]


def test_a_clip_without_admissible_frames_is_an_error():
    with pytest.raises(ValueError):
        nearest_admissible_frame(torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0]), torch.tensor([2]), tolerance=0.1)
