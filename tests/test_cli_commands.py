from fifa2026 import cli

def test_cli_dispatch_table_has_all_commands():
    # main() builds a dispatch with real handlers (not the old print-stub)
    assert hasattr(cli, "_cmd_train") and hasattr(cli, "_cmd_predict")
    assert hasattr(cli, "_cmd_data") and hasattr(cli, "_cmd_evaluate")
