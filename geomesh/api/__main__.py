
from flask import Flask
from localtileserver.tileserver.blueprint import cache, tileserver

app = Flask(__name__, static_url_path='/geomesh-api')
cache.init_app(app)
app.register_blueprint(tileserver, url_prefix='/lts/')

# from flask import Flask, send_file

# app = Flask(__name__, )

# @app.route('/tiles/<til>/<zoom>/<y>/<x>', methods=['GET', 'POST'])
# def tiles(zoom, y, x):
#     default = '_path_to_default_tile\\tiles\\0\\11\\333\\831.png' # this is a blank tile, change to whatever you want
#     filename = '_path_to_tiles\\tiles\\0\\%s\\%s\\%s.png' % (zoom, x, y)
#     if os.path.isfile(filename):
#         return send_file(filename)
#     else:
#         return send_file(default)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return app.send_static_file('index.html')

def main():
    app.run()

if __name__ == "__main__)":
    main()