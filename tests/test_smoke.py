import adamixture
import pytest

def test_import():
    """Prueba simple para ver si el paquete se importa bien"""
    assert adamixture.__version__ is not None

def test_entry_point_help(capsys):
    """Prueba que el comando adamixture --help no explote"""
    from adamixture.entry import main
    import sys
    
    sys.argv = ["adamixture", "--help"]
    
    with pytest.raises(SystemExit):
        main()
    
    captured = capsys.readouterr()
    assert "usage: adamixture" in captured.out or "usage: adamixture" in captured.err