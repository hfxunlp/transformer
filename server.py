#encoding: utf-8

import json
from flask import Flask, render_template, request, send_from_directory

from datautils.bpe import BPEApplier, BPERemover
from datautils.moses import SentenceSplitter
# import Tokenizer/Detokenizer/SentenceSplitter from datautils.zh for Chinese
from datautils.pymoses import Detokenizer, Detruecaser, Normalizepunctuation, Tokenizer, Truecaser
from modules.server.transformer import Translator, TranslatorCore

import cnfg.base as cnfg

"""
slang = "de"# source language
tlang = "en"# target language
tcmodel = "path/to/truecase/model/source language"
tmodel = "path/to/translation/model"
srcvcb = "path/to/source/vocabulary"
tgtvcb = "path/to/target/vocabulary"
bpecds = "path/to/source/bpe/codes"
bpevcb = "path/to/source/bpe/vocabulary"
bpethr = 8# bpe threshold
"""

spl = SentenceSplitter(slang)
tok = Tokenizer(slang)
detok = Detokenizer(tlang)
punc_norm = Normalizepunctuation(slang)
truecaser = Truecaser(tcmodel)
detruecaser = Detruecaser()
tran_core = TranslatorCore(tmodel, srcvcb, tgtvcb, cnfg)
bpe = BPEApplier(bpecds, bpevcb, bpethr)
debpe = BPERemover()
trans = Translator(tran_core, spl, tok, detok, bpe, debpe, punc_norm, truecaser, detruecaser)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def translate_form():

	return render_template("translate.html")

@app.route("/", methods=["POST"])
def translate_core():

	srclang = request.form["src"]

	return render_template("translate.html", tgt=trans(srclang), src=srclang)

@app.route("/api", methods=["POST"])
def translate_core_api():

	try:
		srclang = json.loads(request.get_data())["src"]

	except Exception as e:
		return json.dumps({"exception": str(e)})

	return json.dumps({"tgt": trans(srclang)})

# send everything from client as static content
@app.route("/favicon.ico")
def favicon():

	return send_from_directory(app.root_path, "favicon.ico", mimetype="image/vnd.microsoft.icon")

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8888, debug=False, threaded=True, use_reloader=False, use_debugger=False, use_evalex=False)
