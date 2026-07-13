import json
import os
import tempfile
import unittest

from database import db


class AdminDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.temp_dir.name, 'test_admin.db')
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_admin_dashboard_stats_are_aggregated_from_existing_data(self):
        user_id = db.create_user('admin@example.com', 'hashed', 'Admin User')
        dataset_id = db.save_dataset(user_id, 'sample.csv', 'sample.csv', '/tmp/sample.csv', 100, 5, 2048)

        db.save_preprocessing_result(
            dataset_id,
            user_id,
            json.dumps({
                'best_model': {
                    'metrics': {'accuracy': 0.91}
                }
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        db.save_preprocessing_result(
            dataset_id + 1,
            user_id,
            json.dumps({
                'best_model': {
                    'metrics': {'accuracy': 0.87}
                }
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        db.save_prediction_history(user_id, dataset_id, 'sample.csv', 'Admin User', 'RandomForest', 'yes', 0.91)
        db.save_prediction_history(user_id, dataset_id, 'sample.csv', 'Admin User', 'RandomForest', 'no', 0.85)
        db.save_prediction_history(user_id, dataset_id, 'sample.csv', 'Admin User', 'LogisticRegression', 'yes', 0.79)

        stats = db.get_admin_dashboard_stats()

        self.assertEqual(stats['total_users'], 1)
        self.assertEqual(stats['total_datasets'], 1)
        self.assertEqual(stats['total_predictions'], 3)
        self.assertEqual(stats['most_used_algorithm'], 'RandomForest')
        self.assertAlmostEqual(stats['best_average_accuracy'], 0.88, places=2)


if __name__ == '__main__':
    unittest.main()
