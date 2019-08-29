from urllib import request
import time

url = "http://172.31.2.10:19080/computeScore/test_online_V3/"
with open('/home/appuser/jc_customer.txt') as f:
    lines = f.readlines()
    for line in lines:
        url1 = url + line
        with request.urlopen(url1) as url_rst:
            data = url_rst.read()
            print('data', data.decode('utf-8'))
        time.sleep(0.05)
