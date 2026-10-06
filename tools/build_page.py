"""Build the project page: inject a decoding trace (examples/compare_decoding.py) into docs/page.template.html.
Writes docs/index.html (standalone, for GitHub Pages) and, with --body, the page body without the document skeleton.
Usage: python tools/build_page.py [--trace docs/traces/qwen2.5-7b.json] [--body out.html]"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compact(d):
    for e in d["examples"]:
        for k in ("fp16", "bitnest"):
            e[k]["times"] = [round(t, 4) for t in e[k]["times"]]
            e[k]["prefill_s"] = round(e[k]["prefill_s"], 4); e[k]["tok_s"] = round(e[k]["tok_s"], 2)
        for r in e["bitnest"]["rounds"]:
            r["t"] = round(r["t"], 4)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default=f"{ROOT}/docs/traces/qwen2.5-7b.json"); p.add_argument("--body", default=None)
    a = p.parse_args()
    data = json.dumps(compact(json.load(open(a.trace))), separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    body = open(f"{ROOT}/docs/page.template.html").read().replace("__TRACE__", data)
    head, rest = body.split("<style>", 1)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "<style>\nhtml{-webkit-text-size-adjust:100%}body{margin:0}img{max-width:100%}[hidden]{display:none!important}\n"
            + rest.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    open(f"{ROOT}/docs/index.html", "w").write(page)
    if a.body:
        open(a.body, "w").write(body)
    print(f"wrote docs/index.html ({len(page)/1024:.0f} KB)" + (f" and {a.body}" if a.body else ""))


if __name__ == "__main__":
    main()
