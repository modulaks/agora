import pytest

from agora.db import DB
from agora.graph import ArgumentGraph
from agora.tools import ToolError


@pytest.fixture
def graph(tmp_path):
    return ArgumentGraph(DB(tmp_path / "t.sqlite3"))


def test_status_pipeline(graph):
    q = graph.add_question("Czy X powoduje Y w warunkach Z?", "system")
    c = graph.add_claim("X powoduje Y w warunkach Z, co widać w danych.", "badacz_a", answers=q, confidence=0.9)
    assert graph.db.get_node(c)["status"] == "proposed"
    assert graph.db.get_node(c)["confidence"] <= 0.3
    graph.add_evidence(c, "Źródło pokazuje wzrost Y po X w warunkach Z.", "badacz_a", url="https://example.org/a")
    assert graph.db.get_node(c)["status"] == "accepted"
    o = graph.object(c, "Źródło nie kontroluje zmiennej W, która też wpływa na Y.", "sceptyk")
    assert graph.db.get_node(c)["status"] == "contested"
    with pytest.raises(ToolError):
        graph.rebut(o, "Nie zgadzam się z własnym zarzutem, bo jednak kontroluje W.", "sceptyk")
    graph.rebut(o, "Źródło kontroluje W w analizie wrażliwości w sekcji 4.", "obronca")
    assert graph.db.get_node(c)["status"] == "accepted"
    graph.rule(o, "sustained", "Analiza wrażliwości nie obejmuje W, obrona myli sekcje.", "sedzia")
    assert graph.db.get_node(c)["status"] == "refuted"
    assert graph.db.get_node(q)["status"] == "in_progress"


def test_duplicates_rejected(graph):
    graph.add_claim("Modele językowe rekombinują istniejącą wiedzę naukową.", "a")
    with pytest.raises(ToolError):
        graph.add_claim("Modele językowe rekombinują istniejącą wiedzę naukową!", "b")
    q = graph.add_question("Czy modele językowe tworzą nową wiedzę naukową?", "a")
    with pytest.raises(ToolError):
        graph.add_question("Czy modele językowe tworzą nową wiedzę naukową, tak naprawdę?", "b", parent_id=q)


def test_inference_confidence(graph):
    c1 = graph.add_claim("Każdy opublikowany przypadek odkrycia z udziałem modelu miał ludzką weryfikację eksperymentalną.", "a")
    c2 = graph.add_claim("Weryfikacja eksperymentalna jest warunkiem koniecznym uznania wyniku za wiedzę naukową.", "a")
    graph.add_evidence(c1, "Przegląd dwunastu opisanych przypadków: w każdym eksperyment wykonali ludzie.", "a", kind="data")
    iid, c3 = graph.infer([c1, c2], "deduction", "Żaden opublikowany przypadek nie pokazuje odkrycia bez udziału ludzi w weryfikacji.", "logik")
    assert graph.db.get_node(iid)["status"] == "pending"
    graph.add_evidence(c2, "Standardy metodologiczne czasopism wymagają replikowalnej weryfikacji wyników.", "a", kind="data")
    assert graph.db.get_node(iid)["status"] == "valid"
    assert graph.db.get_node(c3)["status"] == "accepted"
    assert graph.db.get_node(c3)["confidence"] == pytest.approx(0.6 * 0.95, abs=0.01)
    assert "C1" in graph.digest() or "Tezy" in graph.digest()
    assert "CODEX" in graph.codex()
