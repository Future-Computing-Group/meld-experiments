for spec in "cross-encoder/nli-deberta-v3-base nli" "cross-encoder/nli-deberta-v3-large nli" "cross-encoder/nli-roberta-base nli" "FacebookAI/roberta-large-mnli nli" "cross-encoder/quora-distilroberta-base para" "cross-encoder/quora-roberta-base para" "cross-encoder/quora-roberta-large para" "Intel/bert-base-uncased-mrpc para" "semindan/paws_x_xlm_r_only_en para" "domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector para" "cross-encoder/nli-deberta-v3-small nli"; do
  PYTHONPATH=src .venv/bin/python outputs/tournament/score.py $spec 2>&1 | grep -v -i warn
done
