from flask import Flask, request, render_template
import json
import requests
import socket
import time
from datetime import datetime
import re
import data_cleaning_pipeline as dcp

app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
DATA = []
TIMESTAMP = []


@app.route('/score', methods=['GET', 'POST'])
def score():
    m_zero = 0
    request.method == "POST"
    for _ in range(1):
        # Get url that the user has entered
        url = REGISTER_URL
        r = requests.get(url)
        m = re.search('(?<="object_id": )\w+', r.text)
        if m.group(0) != m_zero:
            DATA.append(json.loads(re.sub(r"\s+", " ", r.text)))
            print(type(json.loads(re.sub(r"\s+", " ", r.text))))
            TIMESTAMP.append(time.time())
            m_zero = m.group(0)
        print(len(DATA))
        time.sleep(2)
        for row in DATA:
            pipeline = dcp.Data_Cleaning(row)
            risk, probability = pipeline.run_pipeline()
            df = pipeline.df
            df['fraud_predictions'] = probability
    return render_template('index.html', risk=risk, probability=probability, object_id=m.group(0))


@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}


def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


if __name__ == '__main__':
    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print("attempting to register %s:%d" % (ip_address, PORT))
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
