path=$1
python nli_eval.py --model_name_or_path ${path} --type webnlg ./data/webnlg.csv output_.json
python nli_eval.py --model_name_or_path ${path} --type webnlg --no-templates ./data/webnlg.csv output_backoff.json
python nli_eval.py --model_name_or_path ${path} --type e2e ./data/webnlg.csv output_e2e.json
python nli_eval.py --model_name_or_path ${path} --type e2e --no-templates ./data/webnlg.csv output_e2e_backoff.json
