"""Reproduz a matriz das páginas 6 a 12 do Slide 2."""

from recommender import UserBasedKNN


RATINGS = {
    "Alice": {"Item1": 5, "Item2": 3, "Item3": 4, "Item4": 4},
    "User1": {"Item1": 3, "Item2": 1, "Item3": 2, "Item4": 3, "Item5": 3},
    "User2": {"Item1": 4, "Item2": 3, "Item3": 4, "Item4": 3, "Item5": 5},
    "User3": {"Item1": 3, "Item2": 3, "Item3": 1, "Item4": 5, "Item5": 4},
    "User4": {"Item1": 1, "Item2": 5, "Item3": 5, "Item4": 2, "Item5": 1},
}


if __name__ == "__main__":
    model = UserBasedKNN(RATINGS, k=2)
    print("Similaridade de cada usuário com Alice:")
    for user_id in ("User1", "User2", "User3", "User4"):
        print(f"  {user_id}: {model.similarity('Alice', user_id):.4f}")
    prediction = model.predict("Alice", "Item5")
    if prediction is not None:
        print(f"Previsão de Alice para Item5: {prediction.score:.4f}")
        for neighbor in prediction.neighbors:
            print(f"  {neighbor.user_id}: nota={neighbor.item_rating:g}, "
                  f"média={neighbor.user_mean:.2f}, "
                  f"similaridade={neighbor.similarity:.4f}")
