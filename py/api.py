from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
import news_lstm as news
import dog_cnn as dog

app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}})

api.add_resource(news.NewsClf, '/news/predict')
api.add_resource(dog.DogClf, '/dog/predict')

if __name__ == '__main__':
    app.run(debug=True, port=2808)
