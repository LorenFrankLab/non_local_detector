def test_version_import():
    import non_local_detector

    assert non_local_detector.__version__ is not None
