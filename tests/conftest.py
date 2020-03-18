import os

import numpy as np
import pytest
import hypothesis

np.seterr(all="warn")


@pytest.fixture
def recording_path(tmpdir):
    rec = tmpdir.join("recording.sql")
    yield rec
    # if os.path.isfile(rec):
    #     os.remove(rec)


hypothesis.settings.register_profile("fast", max_examples=5)
hypothesis.settings.register_profile("debugger", report_multiple_bugs=False)
