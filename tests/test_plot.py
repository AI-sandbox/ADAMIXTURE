from adamixture.src.plot import _require_pillow_image


def test_pillow_is_available_through_matplotlib_dependencies() -> None:
    image_module = _require_pillow_image()

    assert callable(image_module.open)
    assert callable(image_module.new)
