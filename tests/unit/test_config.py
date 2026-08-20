from pathlib import Path

from modelforge.config import Settings


def test_settings_do_not_require_provider_secrets(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    assert not settings.frontier_configured
    assert settings.resolve_path(Path("data")) == tmp_path / "data"


def test_settings_secret_is_not_exposed() -> None:
    settings = Settings(
        frontier_base_url="https://provider.invalid/v1",
        frontier_model="frontier-test",
        frontier_api_key="super-secret",
    )
    assert settings.frontier_configured
    assert "super-secret" not in repr(settings)
