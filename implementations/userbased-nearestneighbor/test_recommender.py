import unittest

from example_slide import RATINGS
from recommender import UserBasedKNN, pearson


class SlideExampleTests(unittest.TestCase):
    def setUp(self):
        self.model = UserBasedKNN(RATINGS, k=2)

    def test_slide_similarities_and_prediction(self):
        self.assertAlmostEqual(self.model.similarity("Alice", "User1"), 0.8528028654)
        self.assertAlmostEqual(self.model.similarity("Alice", "User2"), 0.7071067812)
        self.assertAlmostEqual(self.model.similarity("Alice", "User3"), 0.0)
        self.assertAlmostEqual(self.model.similarity("Alice", "User4"), -0.7921180344)
        prediction = self.model.predict("Alice", "Item5")
        self.assertIsNotNone(prediction)
        self.assertEqual([neighbor.user_id for neighbor in prediction.neighbors],
                         ["User1", "User2"])
        self.assertAlmostEqual(prediction.score, 4.8719798994)

    def test_recommendations_are_unseen_and_sorted(self):
        recommendations = self.model.recommend("Alice", n=5)
        self.assertEqual([result.item_id for result in recommendations], ["Item5"])

    def test_no_valid_neighbor_has_no_prediction(self):
        model = UserBasedKNN({"A": {"I1": 5, "I2": 3, "I3": 4},
                              "B": {"I1": 1, "I2": 5, "I3": 2, "I4": 5}})
        self.assertIsNone(model.predict("A", "I4"))

    def test_pearson_requires_variation(self):
        self.assertEqual(pearson({"a": 3, "b": 3, "c": 3},
                                 {"a": 1, "b": 2, "c": 3}), 0.0)


if __name__ == "__main__":
    unittest.main()
