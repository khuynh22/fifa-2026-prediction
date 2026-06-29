.PHONY: data train evaluate predict app all test
data:      ; .venv/Scripts/python.exe -m fifa2026.cli data
train:     ; .venv/Scripts/python.exe -m fifa2026.cli train
evaluate:  ; .venv/Scripts/python.exe -m fifa2026.cli evaluate
predict:   ; .venv/Scripts/python.exe -m fifa2026.cli predict
app:       ; .venv/Scripts/streamlit run app.py
all:       ; $(MAKE) data && $(MAKE) train && $(MAKE) evaluate && $(MAKE) predict
test:      ; .venv/Scripts/python.exe -m pytest -q
