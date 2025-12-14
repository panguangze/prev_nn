# scripts/get_param_yaml.py
# 通用读取 YAML 参数的小工具，供 Bash/Python 混用（输出JSON或标量）
#!/usr/bin/env python3
import sys, json
try:
    import yaml
except:
    print("ERROR: Please pip install pyyaml", file=sys.stderr); sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: get_param_yaml.py <yaml_file> <key_path> [--json]", file=sys.stderr); sys.exit(1)

yaml_file = sys.argv[1]
key_path = sys.argv[2].split(".")
as_json = (len(sys.argv) > 3 and sys.argv[3] == "--json")

with open(yaml_file) as f:
    d = yaml.safe_load(f)

node = d
for k in key_path:
    if k not in node:
        print("", end="")
        sys.exit(0)
    node = node[k]

if as_json:
    print(json.dumps(node))
else:
    if isinstance(node, (dict, list)):
        print(json.dumps(node))
    else:
        print(node)
