def test_ci_probe(capsys):
    import socket

    print("ci-probe host:", socket.gethostname())
    assert True
