from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
module_dict = dict()
success_str = '{"status": 200,"error": 0,"message": "Success","data": null}'
fail_str = '{"status": 1,"error": 1,"message": "failed","data": null}'


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/test/<name>/<int:age>', methods=['GET'])
def hello_test(name, age):
    if name == 'agp':
        return redirect(url_for('hello_admin'))
    return 'Hello %s your age is %d' % (name, age)


@app.route('/test/admin/<name>', methods=['GET'])
def hello_admin(name):
    return 'Hello admin %s' % name

# 动态加载modulemodule
@app.route('/import/<module_name>')
def auto_import(module_name):
    try:
        lib = __import__(module_name)
        module_dict.setdefault(module_name, lib)
    except Exception as e:
        print(e)
        return fail_str
    return success_str


# 动态调用module
@app.route('/run/<module_name>/<paras>')
def auto_run_module(module_name, paras):
    module_dict.get(module_name).run(paras)
    return success_str


if __name__ == '__main__':
    app.run('127.0.0.1', '5005')
