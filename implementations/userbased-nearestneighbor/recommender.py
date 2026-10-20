"""Filtragem colaborativa baseada em usuários, com notas explícitas de 1 a 5."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


Ratings = dict[str, dict[str, float]]


def load_ratings(path: str | Path) -> Ratings:
    """Lê user_id,item_id,rating; pares ausentes ficam ausentes da matriz."""
    ratings: Ratings = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["user_id", "item_id", "rating"]:
            raise ValueError("O CSV deve ter as colunas user_id,item_id,rating.")
        for line_number, row in enumerate(reader, start=2):
            user_id = row["user_id"].strip()
            item_id = row["item_id"].strip()
            if not user_id or not item_id:
                raise ValueError(f"Linha {line_number}: IDs não podem estar vazios.")
            try:
                rating = float(row["rating"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Linha {line_number}: nota inválida.") from exc
            if not math.isfinite(rating) or not 1 <= rating <= 5:
                raise ValueError(f"Linha {line_number}: nota fora da escala de 1 a 5.")
            user_ratings = ratings.setdefault(user_id, {})
            if item_id in user_ratings:
                raise ValueError(f"Linha {line_number}: par usuário-item repetido.")
            user_ratings[item_id] = rating
    if not ratings:
        raise ValueError("O CSV não contém avaliações.")
    return ratings


def pearson(a: Mapping[str, float], b: Mapping[str, float], min_common: int = 3) -> float:
    """Correlação sobre itens em comum; retorna 0 quando não é calculável."""
    if min_common < 2:
        raise ValueError("min_common deve ser pelo menos 2.")
    common = a.keys() & b.keys()
    if len(common) < min_common:
        return 0.0
    a_mean = sum(a[item] for item in common) / len(common)
    b_mean = sum(b[item] for item in common) / len(common)
    numerator = sum((a[item] - a_mean) * (b[item] - b_mean) for item in common)
    a_squared = sum((a[item] - a_mean) ** 2 for item in common)
    b_squared = sum((b[item] - b_mean) ** 2 for item in common)
    if a_squared == 0 or b_squared == 0:
        return 0.0
    return numerator / math.sqrt(a_squared * b_squared)


@dataclass(frozen=True)
class Neighbor:
    user_id: str
    similarity: float
    item_rating: float
    user_mean: float


@dataclass(frozen=True)
class Prediction:
    item_id: str
    score: float
    neighbors: tuple[Neighbor, ...]


class UserBasedKNN:
    """Consulta a matriz de avaliações diretamente; não há ajuste de parâmetros."""

    def __init__(self, ratings: Mapping[str, Mapping[str, float]], k: int = 2,
                 min_common: int = 3) -> None:
        if k < 1:
            raise ValueError("k deve ser pelo menos 1.")
        if min_common < 2:
            raise ValueError("min_common deve ser pelo menos 2.")
        self.ratings = {user: dict(items) for user, items in ratings.items()}
        if not self.ratings or any(not items for items in self.ratings.values()):
            raise ValueError("Cada usuário precisa ter pelo menos uma avaliação.")
        self.k = k
        self.min_common = min_common
        self.user_means = {
            user: sum(items.values()) / len(items)
            for user, items in self.ratings.items()
        }
        self.items = set().union(*(items.keys() for items in self.ratings.values()))

    def similarity(self, user_a: str, user_b: str) -> float:
        return pearson(self.ratings[user_a], self.ratings[user_b], self.min_common)

    def predict(self, user_id: str, item_id: str) -> Prediction | None:
        """Prevê item não avaliado; None indica ausência de vizinhos válidos."""
        user_ratings = self.ratings[user_id]
        if item_id in user_ratings:
            raise ValueError("O usuário já avaliou este item.")
        neighbors = []
        for other_id, other_ratings in self.ratings.items():
            if other_id == user_id or item_id not in other_ratings:
                continue
            similarity = self.similarity(user_id, other_id)
            if similarity > 0:
                neighbors.append(Neighbor(
                    other_id, similarity, other_ratings[item_id],
                    self.user_means[other_id],
                ))
        neighbors.sort(key=lambda neighbor: (-neighbor.similarity, neighbor.user_id))
        selected = tuple(neighbors[:self.k])
        if not selected:
            return None
        adjustment = sum(
            neighbor.similarity * (neighbor.item_rating - neighbor.user_mean)
            for neighbor in selected
        ) / sum(neighbor.similarity for neighbor in selected)
        score = max(1.0, min(5.0, self.user_means[user_id] + adjustment))
        return Prediction(item_id, score, selected)

    def recommend(self, user_id: str, n: int = 5) -> list[Prediction]:
        """Devolve os n itens não avaliados com maior nota prevista."""
        if n < 1:
            raise ValueError("n deve ser pelo menos 1.")
        seen = self.ratings[user_id]
        predictions = [self.predict(user_id, item) for item in self.items - seen.keys()]
        available = [prediction for prediction in predictions if prediction is not None]
        return sorted(available, key=lambda prediction: (-prediction.score, prediction.item_id))[:n]
