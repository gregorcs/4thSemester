import unittest

from flask import json
from ..UserInterface import app

class TestSmokingPredictionService(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_make_prediction(self):
        test_data = {
            'age': 25,
            'body_type': 'average',
            'diet': 'mostly anything',
            'drinks': 'socially',
            'drugs': 'never',
            'education': 'graduated from college/university',
            'ethnicity': 'white',
            'income': -1,
            'job': 'student',
            'location': 'san francisco, california',
            'offspring': 'doesnt have kids, but might want them',
            'orientation': 'straight',
            'religion': 'agnosticism and very serious about it',
            'sex': 'm'
        }
        
        response = self.app.post('/predict', 
                                 data=json.dumps(test_data), 
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)

if __name__ == '__main__':
    unittest.main()
