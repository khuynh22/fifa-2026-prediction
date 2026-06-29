import os
from fifa2026 import cli

def test_cli_dispatch_table_has_all_commands():
    # main() builds a dispatch with real handlers (not the old print-stub)
    assert hasattr(cli, "_cmd_train") and hasattr(cli, "_cmd_predict")
    assert hasattr(cli, "_cmd_data") and hasattr(cli, "_cmd_evaluate")


def test_load_dotenv_sets_missing_and_respects_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFIFA_TEST_KEY=from_file\nFIFA_EXISTING=from_file\n", encoding="utf-8")
    monkeypatch.delenv("FIFA_TEST_KEY", raising=False)
    monkeypatch.setenv("FIFA_EXISTING", "from_env")
    cli.load_dotenv(env)
    assert os.environ["FIFA_TEST_KEY"] == "from_file"      # missing -> set from file
    assert os.environ["FIFA_EXISTING"] == "from_env"        # existing env wins (setdefault)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    cli.load_dotenv(tmp_path / "nope.env")  # must not raise
