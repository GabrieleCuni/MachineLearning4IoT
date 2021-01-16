import cherrypy
import json

class Calculator:
    exposed = True

    def GET(self, *path, **query):
        if len(path) != 1:
            raise cherrypy.HTTPError(400, "Wrong path")
        if len(query) != 2:
            raise cherrypy.HTTPError(400, "Wrong query")
        operation = path[0]
        op1 = query.get("op1")
        if op1 is None:
            raise cherrypy.HTTPError(400, "Op1 missing")
        else:
            op1 = float(op1)
        op2 = query.get("op2")
        if op2 is None:
            raise cherrypy.HTTPError(400, "Op2 missing")
        else:
            op2 = float(op2)
        if operation == "add":
            result = op1 + op2
        elif operation == "sub":
            result = op1 - op2
        elif operation == "mul":
            result = op1 * op2
        elif operation == "div":
            if op2 == 0:
                raise cherrypy.HTTPError(400, "Can not divide per zero")
            result = op1 / op2
        else:
            raise cherrypy.HTTPError(400, "Wrong operation")
        output = {"command":operation,"op1":op1,"op2":op2,"result":result}
        output_json = json.dumps(output)
        return output_json

    def PUT(self, *path, **query):
        body_string = cherrypy.request.body.read()
        body_dict = json.loads(body_string)
        operation = body_dict["command"]
        op_list = body_dict["operands"]

        if operation == "add":
            result = 0
            for op in op_list:
                result += op
        elif operation == "sub":
            result = op_list[0]
            tmp_list = op_list
            tmp_list.remove(result)
            for op in tmp_list:
                result -= op
        elif operation == "mul":
            result = op_list[0]
            tmp_list = op_list
            tmp_list.remove(result)
            for op in tmp_list:
                result *= op
        elif operation == "div":
            result = op_list[0]
            tmp_list = op_list
            tmp_list.remove(result)
            for op in tmp_list:
                if op == 0:
                    raise cherrypy.HTTPError(400, "Can not divide per zero")
                result /= op
        else:
            raise cherrypy.HTTPError(400, "Wrong operation")
        output = {"command":operation,"op":op_list,"result":result}
        output_json = json.dumps(output)
        return output_json

    def POST(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

def main():
    conf = {"/":{"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), "", conf)
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.config.update({"server.socket_port": 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()    


if __name__ == "__main__":
    main()