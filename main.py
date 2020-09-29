from flask import Flask, jsonify, request, render_template, send_from_directory, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float
from flask_marshmallow import Marshmallow
import os
import time
from datetime import datetime as dt
import csv
import pytz
import pandas as pd
import requests
from Predict import GroupLSTM
from Predict import GetARIMA
from threading import Thread

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.db')
db = SQLAlchemy(app)
ma = Marshmallow(app)


# database models
class Sensors(db.Model):
    __tablename__ = 'sensors'
    id = Column(Integer, primary_key=True)
    datetime = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    heat_index = Column(Float)


class LSTMData(db.Model):
    __tablename__ = 'lstmData'
    id = Column(Integer, primary_key=True)
    datetime = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    heat_index = Column(Float)


class ARIMAData(db.Model):
    __tablename__ = 'ARIMAData'
    id = Column(Integer, primary_key=True)
    datetime = Column(String)
    temperature = Column(Float)
    humidity = Column(Float)
    heat_index = Column(Float)


class SensorSchema(ma.Schema):
    class Meta:
        fields = ('id', 'datetime', 'temperature', 'humidity', 'heat_index')


class LSTMSchema(ma.Schema):
    class Meta:
        fields = ('id', 'datetime', 'temperature', 'humidity', 'heat_index')


class ARIMASchema(ma.Schema):
    class Meta:
        fields = ('id', 'datetime', 'temperature', 'humidity', 'heat_index')


sensor_schema = SensorSchema()
sensors_schema = SensorSchema(many=True)

lstm_schema = LSTMSchema()
lstm_schemas = LSTMSchema(many=True)

arima_schema = ARIMASchema()
arima_schemas = ARIMASchema(many=True)


def get_db_data(table):
    tables = {'sensor': [Sensors, sensors_schema], 'lstm': [LSTMData, lstm_schemas],
              'arima': [ARIMAData, arima_schemas]}
    return tables[table][1].dump(tables[table][0].query.all())


lstm_agent = GroupLSTM(data=get_db_data('lstm'))
arima_agent = GetARIMA(data=get_db_data('arima'))


@app.cli.command('db_create')
def db_create():
    db.create_all()
    print('database created!')


@app.cli.command('db_drop')
def db_drop():
    db.drop_all()
    print('database dropped!')


@app.cli.command('db_seed')
def db_seed():
    data = [20.26, 40.11, 20.55]
    funcs = [Sensors, LSTMData, ARIMAData]
    for table in funcs:
        entry = table(datetime="{:%d-%m-%Y %H:%M:%S}".format(dt.now()),
                      temperature=data[0],
                      humidity=data[1],
                      heat_index=data[2])
        time.sleep(1)
        db.session.add(entry)
    # first = Sensors(datetime="{:%d-%m-%Y %H:%M:%S}".format(dt.now()),
    #                 temperature=20.26,
    #                 humidity=14.11)
    # time.sleep(1)
    # second = Sensors(datetime="{:%d-%m-%Y %H:%M:%S}".format(dt.now()),
    #                  temperature=21.42,
    #                  humidity=15.11)
    # time.sleep(1)
    # third = Sensors(datetime="{:%d-%m-%Y %H:%M:%S}".format(dt.now()),
    #                 temperature=22.33,
    #                 humidity=13.41)

    # db.session.add(first)
    # db.session.add(second)
    # db.session.add(third)
    db.session.commit()
    print('database seeded!')


# @app.cli.command('db_save')
def save_data():
    london = pytz.timezone('Europe/London')
    folder = "static/csv_data"
    files = os.listdir(folder)
    if len(files) > 7:
        files.sort()
        os.remove(f"{folder}/{files[0]}")
    path_name = f'{folder}/{"{:%d %b %Y}".format(dt.now().astimezone(london))}.csv'
    with open(path_name, 'w', newline='\n') as f:
        out = csv.writer(f)
        out.writerow(['id', 'datetime', 'temperature', 'humidity', 'heat_index', 'lstm_temp', 'lstm_hum', 'lstm_heat',
                      'arima_temp', 'arima_hum', 'arima_heat'])
        s_data = db.session.query(Sensors).all()
        l_data = db.session.query(LSTMData).all()
        a_data = db.session.query(ARIMAData).all()

        for i in range(len(s_data)):
            row = [s_data[i].id, s_data[i].datetime, s_data[i].temperature, s_data[i].humidity, s_data[i].heat_index,
                   l_data[i].temperature, l_data[i].humidity, l_data[i].heat_index,
                   a_data[i].temperature, a_data[i].humidity, a_data[i].heat_index]
            out.writerow(row)
        out.writerow(['id', 'datetime', 'temperature', 'humidity'])

        # for item in db.session.query(Sensors).all():
        #     row = [item.id, item.datetime, item.temperature, item.humidity]
        #     out.writerow(row)


# @app.cli.command('db_test')
def delete_rows():
    tables = [Sensors, LSTMData, ARIMAData]
    for table in tables:
        obj = db.session.query(table).order_by(table.id.desc()).first()
        all_data = Sensors.query.limit(obj.id - 1).all()
        for row in all_data:
            db.session.delete(row)
    db.session.commit()
    print('rows deleted!')


@app.route('/')
def hello_world():
    return jsonify({'message': 'welcome to api interface'})


def add_data(temperature, humidity):
    # my_date, heat_index, arima_temp, arima_hum, arima_heat, lstm_temp, lstm_hum, lstm_heat):

    london = pytz.timezone('Europe/London')
    time_now = dt.now().astimezone(london)
    raw_save_time = '23:59:45'
    save_time = [int(i) for i in raw_save_time.split(':')]
    if (time_now.hour == save_time[0]) and (time_now.minute == save_time[1]) and (time_now.second >= save_time[2]):
        if f'{"{:%d %b %Y}".format(dt.now().astimezone(london))}.csv' not in os.listdir('static/csv_data'):
            print('\n\nsaving data\n\n ')
            save_data()
            delete_rows()

    heat_index = get_heat_index(temperature, humidity)
    my_date = "{:%d-%m-%Y %H:%M:%S}".format(dt.now().astimezone(london))
    arima_agent.data = get_db_data('arima')
    arima_data = arima_agent.predict()
    lstm_data = lstm_agent.predict(get_db_data('lstm'))

    data1 = Sensors(datetime=my_date,
                    temperature=temperature,
                    humidity=humidity,
                    heat_index=heat_index)

    data2 = LSTMData(datetime=my_date,
                     temperature=lstm_data['temp'],
                     humidity=lstm_data['hum'],
                     heat_index=lstm_data['heat'])

    data3 = ARIMAData(datetime=my_date,
                      temperature=arima_data['temp'],
                      humidity=arima_data['hum'],
                      heat_index=arima_data['heat'])

    db.session.add(data1)
    db.session.add(data2)
    db.session.add(data3)
    db.session.commit()


def send_data_to_server():
    data = get_data()
    endpoint = 'https://lsbu-sensors.herokuapp.com/send'
    requests.post(endpoint, json=data, )


@app.route('/send')
def send_data():
    try:
        add_data(temperature=float(request.args.get('temperature')), humidity=float(request.args.get('humidity')))
        t1 = Thread(target=send_data_to_server)
        t1.start()
        # add_data(temperature=float(request.args.get('temperature')), humidity=float(request.args.get('pressure')),
        #          my_date=request.args.get('datetime'), heat_index=float(request.args.get('heat_index')),
        #          lstm_heat=float(request.args.get('lstm_heat')), lstm_hum=float(request.args.get('lstm_hum')),
        #          lstm_temp=float(request.args.get('lstm_temp')), arima_heat=float(request.args.get('arima_heat')),
        #          arima_temp=float(request.args.get('arima_temp')), arima_hum=float(request.args.get('arima_hum')))
        return jsonify({'info': 'data received'}), 200
    except ValueError:
        return jsonify({'info': 'Value Error! floats only!'}), 400


@app.route("/download", methods=["POST", "GET"])
def get_csv():
    try:
        return send_from_directory('static/csv_data', filename=request.form["myfile"], as_attachment=True)
    except FileNotFoundError:
        abort(404)


# @app.route("/get-data")
# def get_data():
#     obj = db.session.query(Sensors).order_by(Sensors.id.desc()).first()
#     return jsonify({'datetime': obj.datetime, 'temperature': obj.temperature, 'humidity': obj.humidity}), 200

def get_data():
    actual = {}
    for table in ['sensor', 'lstm', 'arima']:
        actual[table] = get_db_data(table)[-1]
    new_stat = sensor_stat.get_stat()
    pred_stat = {'lstm': lstm_agent.describe(), 'arima': arima_agent.get_stat()}
    result = {'actual': actual, 'data_stat': new_stat, 'pred_stat': pred_stat}
    return result


def stat():
    result = sensors_schema.dump(Sensors.query.all())
    df = pd.DataFrame(result)
    return df.describe()


@app.route("/describe")
def get_stat():
    dfb = stat()
    return dfb.to_json()


@app.route("/sensor-data/<int:length>")
def sensor_data(length=50):
    result = {}
    for table in ['sensor', 'lstm', 'arima']:
        result[table] = get_db_data(table)[-length:]

    # result = sensors_schema.dump(Sensors.query.all())[:length]
    return jsonify(result), 200


def lru_cache():
    folder = "static/temp"
    files = os.listdir(folder)
    max_len = 20
    if len(files) > max_len:
        files.sort()
        os.remove(f"{folder}/{files[0]}")


def get_heat_index(temperature, humidity):
    cons = {1: {'c1': -8.78469475556,
                'c2': 1.61139411,
                'c3': 2.33854883889,
                'c4': -0.14611605,
                'c5': -0.012308094,
                'c6': -0.0164248277778,
                'c7': 0.002211732,
                'c8': 0.00072546,
                'c9': -0.000003582}}

    funcs = {1: lambda t, h: cons[1]['c1'] + (cons[1]['c2'] * t) + (cons[1]['c3'] * h) + (cons[1]['c4'] * h * t) +
                             (cons[1]['c5'] * t ** 2) + (cons[1]['c6'] * h ** 2) + (cons[1]['c7'] * t ** 2 * h) +
                             (cons[1]['c8'] * t * h ** 2) + (cons[1]['c9'] * (t ** 2) * (h ** 2)),
             }
    return funcs[1](temperature, humidity)


@app.route("/sensor-data/csv/<int:length>")
def sensor_data_csv(length=50):
    folder = "static/temp"
    file_name = f'{int(time.time())}.csv'
    lru_cache()
    with open(f'{folder}/{file_name}', 'w', newline='\n') as f:
        out = csv.writer(f)
        out.writerow(['id', 'datetime', 'temperature', 'humidity', 'heat_index', 'lstm_temp', 'lstm_hum', 'lstm_heat',
                      'arima_temp', 'arima_hum', 'arima_heat'])
        s_data = db.session.query(Sensors).all()[-length:]
        l_data = db.session.query(LSTMData).all()[-length:]
        a_data = db.session.query(ARIMAData).all()[-length:]
        for i in range(len(s_data)):
            row = [s_data[i].id, s_data[i].datetime, s_data[i].temperature, s_data[i].humidity, s_data[i].heat_index,
                   l_data[i].temperature, l_data[i].humidity, l_data[i].heat_index,
                   a_data[i].temperature, a_data[i].humidity, a_data[i].heat_index]
            out.writerow(row)
        # for item in db.session.query(Sensors).all()[-length:]:
        #     row = [item.id, item.datetime, item.temperature, item.humidity]
        #     out.writerow(row)

    return send_from_directory(folder, filename=file_name, as_attachment=True)


# db_stat = stat()


class DataStat:
    """:key
    {'temperature': {'count': 3001.0, 'mean': 25.1823592136, 'std': 0.39777284, 'min': 20.26, '25%': 25.0, '50%': 25.0, '75%': 25.0, 'max': 26.0},
    'id': {'count': 3001.0, 'mean': 1501.0, 'std': 866.4584044642, 'min': 1.0, '25%': 751.0, '50%': 1501.0, '75%': 2251.0, 'max': 3001.0},
    'humidity': {'count': 3001.0, 'mean': 39.0640153282, 'std': 1.1322446902, 'min': 36.0, '25%': 38.0, '50%': 39.0, '75%': 40.0, 'max': 42.0},
    'heat_index': {'count': 3001.0, 'mean': 25.7518516255, 'std': 0.2332050521, 'min': 20.55, '25%': 25.6297423111, '50%': 25.6834783556, '75%': 25.6834783556, 'max': 26.2747953298}}

    """

    def __init__(self):
        self.stat = stat()
        self.units = ['temperature', 'humidity', 'heat_index']
        self.metrics = ['count', 'mean', 'std', 'min', 'max']

    @staticmethod
    def percentage(new_value, old_value):
        return round((abs(new_value - old_value) / old_value) * 100, 2)

    @staticmethod
    def get_arrow(new_value, old_value):
        arrow = 'down'
        if new_value > old_value:
            arrow = 'up'
        elif new_value == old_value:
            arrow = 'equal'
        return arrow

    def get_stat(self):
        new_stat = stat()
        # new_stat[i]
        return {i: {j: {'data': new_stat[i][j], 'arrow': self.get_arrow(new_stat[i][j], self.stat[i][j]),
                        '%': self.percentage(new_stat[i][j], self.stat[i][j])} for j in self.metrics} for i in
                self.units}


# obj = DataStat()
#
# print(obj.get_stat())

sensor_stat = DataStat()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
