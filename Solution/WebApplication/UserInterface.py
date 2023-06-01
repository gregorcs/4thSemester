from flask import Flask, render_template

from Endpoints import endpoints

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


app.register_blueprint(endpoints)

if __name__ == '__main__':
    app.run(debug=True)
