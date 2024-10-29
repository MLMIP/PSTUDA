from wsgiref.simple_server import make_server
import os
import sys
import numpy as np

logDir, port = '', 1998
losshtml = [
    "<!DOCTYPE html> <html lang=\"zh-CN\" style=\"height: 100%\"> <head><meta charset=\"utf-8\"> "
    "</head> <body style=\"height: 100%; margin: 0\">",
    "<script type = \"text/javascript\" src = \"https://fastly.jsdelivr.net/npm/echarts@5.3.2/dist/echarts.min.js\" > "
    "</script > <script type = \"text/javascript\" >",
    "</script> </body> </html >"]
Canvas = [
    "var myChart = echarts.init(document.getElementById('",
    "'), null, {renderer: 'canvas',useDirtyRect: false});option = {title: {text: '",
    "'},tooltip: {trigger: 'axis'},legend: {data: ['Value', 'Smooth']},animation: false,grid: {top: 40,left: 50,"
    "right: 40,bottom: 50},color:['#33a3dc','#fc8452',],xAxis: {name: 'x',minorTick: {show: true},minorSplitLine: {"
    "show: true}},yAxis: {name: 'y',min: 0,max: ",
    ",minorTick: {show: true},minorSplitLine: {show: true}},dataZoom: ["
    "{show: true,type: 'inside',filterMode: 'none',xAxisIndex: [0],startValue: 0},{show: true,"
    "type: 'inside',filterMode: 'none',yAxisIndex: [0],startValue: 0}],series: [{type: 'line',"
    "showSymbol: false,clip: true,name:'Value',data: ",
    "},{type: 'line',showSymbol: false,clip: true,name:'Smooth',data:",
    "}],};myChart.setOption(option);window.addEventListener('resize', myChart.resize);\n"]


def app(env, start_response):
    Index = env['PATH_INFO'].rfind('/') + 1
    Prefix, Suffix = env['PATH_INFO'][:Index], env['PATH_INFO'][Index:]
    print(Prefix + "\t\t" + Suffix)
    if Suffix[-5:] == '.html':
        start_response("200 ok", [("Content-Type", "text/html")])
        with open(logDir + env['PATH_INFO'], "rb") as f:
            a = f.read()
        return [a]
    if Suffix[-4:] == '.txt' or Suffix[-5:] == '.json' or Suffix[-4:] == '.log':
        start_response("200 OK", [('Content-Type', 'text/plain; charset=utf-8')])
        with open(logDir + env['PATH_INFO'], "rb") as f:
            a = f.read()
        return [a]
    if Suffix[-3:] == '.py':
        start_response("200 OK", [('Content-Type', 'text/text; charset=utf-8')])
        with open(logDir + env['PATH_INFO'], "rb") as f:
            a = f.read()
        return [a]
    if Suffix[-4:] == '.pth' or Suffix[-5:] == '.ckpt' or Suffix[-4:] == '.pkl':
        start_response("200 OK", [('Content-Type', 'html/text; charset=utf-8')])
        with open(logDir + env['PATH_INFO'], "rb") as f:
            a = f.read()
        return [a]
    if Suffix[-4:] == '.jpg' or Suffix[-4:] == '.png':
        start_response("200 ok", [("Content-Type", "image/webp")])
        with open(logDir + env['PATH_INFO'], 'rb') as f:
            img = f.read()
        return [img]
    if Suffix == 'Loss':
        print(logDir + Prefix + "loss.txt")
        data = np.loadtxt(logDir + Prefix + "loss.txt")
        if len(data.shape) == 1:
            data = data.reshape([data.shape[0],1])
        with open(logDir + Prefix + "LossName.txt", 'r') as f:
            lossName = f.read()
        html = losshtml
        lossName = lossName.split(' ')
        Html = html[0] + "\n"
        for lName in lossName:
            Html += "<div id = \"{}\" style = \"height: 50%;width: 48%;float: left;\"> </div>\n".format(lName)
        Html += html[1] + "\n"
        for index, lName in enumerate(lossName):
            if data.shape[0] > 200:
                ol = min(int(data.shape[0] / 100), 200)
                point = np.convolve(data[:, index], np.ones(ol))[ol:-ol] / ol
            else:
                point = data[:, index]
            step = len(point) // 5000 + 1
            Html += Canvas[0] + lName + Canvas[1] + lName + Canvas[2] + str(min(max(data[:, index]) * 1.5, 10)) + \
                    Canvas[3] + [[i, data[i, index]] for i in range(0, data.shape[0], step)].__str__() + \
                    Canvas[4] + [[i, point[i]] for i in range(0, len(point), step)].__str__() + Canvas[5]
        Html += html[2] + "\n"
        start_response("200 ok", [("Content-Type", "text/html")])
        return [bytes(Html, encoding='UTF-8')]

    if os.path.isdir(logDir + env['PATH_INFO']):
        start_response("200 ok", [("Content-Type", "text/html")])
        page = "<!DOCTYPE html> <html> <head><meta charset=\"utf-8\"> <title>服务器文件</title> <style> 	body{ background: " \
               "#f6f5ec; text-align: center; } .icoimg { margin-bottom: -5px; margin-right: 5pt; width: 25pt; } .item { " \
               "border: 0pt thick double #32a1ce; background: white; margin: 1px 10% 1px; text-align: justify; display: " \
               "block; } .itemIn{ margin: 0pt 2% 0pt; } </style> </head> <body><h1>目录</h1> "
        filename = sorted(os.listdir(logDir + env['PATH_INFO']))
        if filename.count('LossName.txt') > 0:
            page += "<div class =\"item\" ><a href=\"" + env['PATH_INFO'] + "/Loss" + "\"><div class =\"itemIn\"><h2>Loss</h2></div></a></div >"
        FileFolder, Files = '', ''
        if env['PATH_INFO'][-1] != '/':
            env['PATH_INFO'] += '/'
        for name in filename:
            if name[0] == '_' or name[0] == '.':
                continue

            pre = "<div class=\"item\" ><a href=\"" + env['PATH_INFO'] + name
            mid = "\"><div class=\"itemIn\"> <h2><img src=\""
            suf = "\"class=\"icoimg\"/>" + name + "</h2> </div> </a></div> "
            if os.path.isdir(logDir + env['PATH_INFO'] + name):
                FileFolder += pre + mid + os.getcwd() + "/a.jpg" + suf
                continue
            if name[name.rfind('.') + 1:] in ['txt', 'json', 'html', 'py', 'pdf', 'png', 'jpg','pth', 'ckpt', 'pkl', 'log', 'pt']:
                Files += pre + mid +  os.getcwd() + "/b.jpg" + suf
        page += FileFolder + Files + "</body></html>"
        return [bytes(page, encoding='UTF-8')]

    start_response("200 ok", [("Content-Type", "text/html")])
    return [bytes('404', encoding='UTF-8')]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        logDir = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    server = make_server("", port, app)
    print("127.0.0.1:{}\t".format(port, logDir))
    server.serve_forever()
